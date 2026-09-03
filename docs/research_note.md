# autoresearch: NanoGPT speedrun on RTX 5090 x1 — 結果ノート

目的: `val_loss <= 3.28` 到達までの `train_time` を短縮する（1x RTX 5090, 32GB, sm_120）。

## 現在の最小 train_time

| 日付 | train_time | val_loss | log | 変更 |
|---|---|---|---|---|
| 2026-09-02 | 1703.7 s | 3.2805 | `logs/c0f0c03b-9c5b-489a-a163-c4fff78b2518.txt` | baseline (H100x8 コードをそのまま RTX5090x1 で実行) |
| 2026-09-02 | 1429.8 s | 3.2779 | `logs/9e0fc6e7-a0de-4a99-a7dc-454fcea70fd6.txt` | Exp 1: DC attention backward カーネル書き直し (−16.1%) |
| 2026-09-03 | 1304.9 s | 3.2772 | `logs/f87cde40-8371-4e6a-bdad-25918515eb75.txt` | Exp 2b: embedding 行勾配 + micro-batch 4/8/12 + MLP カーネル設定 (−8.7%) |
| 2026-09-03 | **1210.5 s** | 3.2763 | `logs/a8e283c2-215d-4e87-a39a-5d92422e9784.txt` | Exp 3: FP8 MLP backward (−7.2%) |

## 環境
- RTX 5090 (170 SM, 32 GB, 99 KB smem/block), driver 610.57, torch 2.10.0+cu128, triton 3.6.0, cuDNN 9.10
- FlashAttention-3 は sm_120 非対応 → `kernels-community/flash-attn2` にフォールバック
- `grad_accum_steps = 16`（micro-batch 8192 → 16384 → 24576 tokens）、ピークメモリ ~24 GB

## Baseline プロファイル (2026-09-02)
`torch.profiler` で 1 step を計測（stage1: 728 ms/step, stage2: 1326 ms, stage3: 1945 ms）。GPU は飽和（CPU 側の launch queue が "Command Buffer Full"）なので、カーネル効率がすべて。

stage1 (131k tokens/step, 16 micro-batch) の内訳（GPU時間）:

| カテゴリ | ms/step | 備考 |
|---|---|---|
| bf16 GEMM (cuBLAS) | 223 | attention QKVO と MLP backward。~195 TFLOPS ≈ bf16 ピーク(~210) の 93% → 改善余地はFP8化のみ |
| DC attention backward (`_dc_postonly_corr_bwd_pre_wsmall_kernel`) | 112 | **7 ms/micro-batch。forward は 0.1 ms** ← 異常 |
| inductor triton (elementwise/norm/embedding backward) | 103 | |
| fused MLP triton (`linear_relu_square`) | 85 | cuBLAS 相当なら ~50 ms |
| fp8 GEMM (cuBLAS) | 83 | lm_head + MLP down-proj。~440 TFLOPS |
| grad accumulation add (`CUDAFunctor_add`) | 37 | micro-batch ごとの .grad 加算（~1.4 GB のgrad） |
| flash-attn2 | 34 | stage3 では 170 ms (~9%) |
| CE kernel | 12.5 | |
| optimizer (polar express 等) | ~14 | |

cuBLAS 実測ピーク: bf16 200-227 TFLOPS, fp8 400-450 TFLOPS。

## 実験ログ

### Exp 1: DC attention backward カーネルの書き直し（2026-09-02）
**問題**: `dc_triton_kernels.py` の backward "pre" カーネルは 6 head 分の (16x128) 確率タイルと 128x128 の K/V/dV タイルを同時に生存させるため、ptxas がレジスタ割当を諦めて 40 regs + 54 KB/thread のスピル（sm_90a でも 32 regs + 64 KB スピル → H100 でも遅かったはず）。
さらに dV/dK を bf16 の atomic_add で累積していた（精度も悪い）。

**変更**:
- key window (128) を SUB_K=64 のサブブロックに分けてループ、生存タイルを最小化（数式・出力契約は同一）
- dV, dK を fp32 で atomic 累積 → 最後に bf16 へ変換
- num_warps 8→8, SUB_K=64（sweep: 16/32/64 x 2/4/8 warps はどれも ±5%）

**マイクロベンチ** (fwd+bwd, H=6, D=128, window=112):

| T | before | after |
|---|---|---|
| 8192 | 6.9 ms | 0.44 ms |
| 16384 | 13.8 ms | 0.91 ms |
| 24576 | 20.7 ms | 1.38 ms |

fp32 リファレンスとの誤差: q/w1/w2 grad は同等、k/v grad は改善（rms 相対誤差 4.3e-3 → 2.7e-3）。

**step time** (bench harness, 3 step 平均): stage1 728→620 ms, stage2 1326→1112 ms, stage3 1945→1631 ms（−15〜16%）。

**結果**: `logs/9e0fc6e7-a0de-4a99-a7dc-454fcea70fd6.txt` — **train_time 1429.8 s, val_loss 3.2779**（baseline 1703.7 s / 3.2805）。−16.1%。途中の val_loss も baseline と一致（step250: 4.4988 vs 4.5053, step1000: 3.4163 vs 3.4190）。ピークメモリ 23.9 GB（変化なし）。

codex による差分レビュー: 実バグなし（window>113 の pre-existing な制限を指摘、window=112 なので無関係）。NaN 安全のため `a_hidden` の valid マスクを復元して commit。

### Exp 2: 純粋な速度改善 4 点セット（2026-09-02）
Exp 1 の上に、数式を変えない 4 つの変更をまとめて検証（それぞれ bench harness で個別に効果確認済み）:

1. **Embedding 勾配の行単位累積** (`gather_embedding_rows` / `accumulate_embedding_grads`):
   autograd の embedding backward は micro-batch ごとに dense な (vocab, dim) 勾配を生成（zero-fill + scatter）し、さらに `.grad` へ全サイズの add をしていた。value_embeds (5x50304x768) + bigram_embed (377280x768) + embed で ~1 GB/micro-batch の無駄なメモリトラフィック。
   行を compiled graph の外で gather して leaf tensor として渡し、(tokens, dim) の行勾配を `index_add_` で `.grad` に足すように変更。
   マイクロベンチ: dense path ~1.5-2 ms/table vs index_add_ 0.03-0.2 ms。
2. **stage ごとの micro-batch 数** (`MAX_MICRO_BATCH_TOKENS=32768` → grad_accum 8/8/16):
   H100x8 の per-GPU micro-batch (16384/32768 tokens) を stage 1-2 で再現。stage 3 (49152 tokens) は OOM するので 16 のまま。
   `grad_scale`（loss スケールと lm_head の FP8 grad scale）を `ForwardScheduleConfig` 経由で stage ごとに渡す。
   bench: stage1 620→549 ms, stage2 1112→1052 ms（この項目単体）。
3. **fused MLP Triton カーネルのタイル設定** (99 KB smem 向け): backward `dpre` は従来 num_stages=1 で動いていた（0.335 ms @T=8192）。BM=128/BN=64/BK=64/3 stages で 0.251 ms（cuBLAS bf16 + epilogue 0.252 ms と同等、T=24576 では 0.715 vs 0.807 で Triton が勝ち）。forward は BK=64/3 stages で ~3% 改善。
4. `FusedSoftcappedCrossEntropy` が backward 用に (rows, vocab) の bf16 logits を保存していたのをやめる（backward では shape しか使っていなかった）。stage 3 の 8 micro-batch 化には足りず OOM のまま。ついでに backward の返り値数（10 入力に対し 9）を修正。

**bench (3 step 平均)**: stage1 620→504 ms, stage2 1112→991 ms, stage3 1631→1529 ms（Exp1 比 −19% / −11% / −6%）。ピーク 21.5 GB。
予測 train_time ≈ 1300 s。

**結果 (失敗)**: 1 回目 (`logs/ca7f77f0-...`) は warmup 中に OOM（warmup 用の model/optimizer state スナップショット ~6 GB が GPU 上にあったため。CPU に置くよう修正）。
2 回目 `logs/949b06dc-dbee-43d4-bc5e-9c5d0a80d124.txt`: train_time 1288 s だが **val_loss が step 250 以降 NaN**。
原因: 3. のタイル設定の編集で `aux = torch.empty(...)`（forward 用ダミー）が `if aux is None:` ブロックの外に出てしまい、backward の `dpre` カーネルの `aux`（= `post`）が未初期化バッファに置き換わっていた（step 1 で NaN）。bench harness は損失を表示していなかったので気付けなかった → harness に LOSS 表示を追加、実験前に必ず 10 step 程度の損失を確認する運用に変更。

**教訓**: 速度だけでなく損失の有限性・軌跡を短い run で必ず確認する。

### Exp 2b: Exp 2 の修正版 + micro-batch 4/8/12 + chunked CE（2026-09-02）
- aux バグ修正
- micro-batch 数の "8 の倍数" 制約を撤廃（codex の指摘）: 32768 tokens/micro-batch で 4/8/12 に
- `FusedSoftcappedCrossEntropy` を 16384 行ずつのチャンク処理に（(rows, vocab) bf16 logits と fp8 勾配の転置を全サイズで持たない）。MTP の先読みはチャンク境界を跨げるよう `n_targets` を別引数に

### 調査: FlashAttention-4 (CuTe DSL, sm_120 カーネル) vs hub flash-attn2（2026-09-02）
`flash-attn-4==4.0.0b29` + `nvidia-cutlass-dsl 4.6.2` を別 venv に入れて varlen + causal + sliding window を比較。
T=32768, maxdoc=896: FA2 fwd 0.242 / fwd+bwd 1.385 ms vs FA4 fwd 0.333 / fwd+bwd 1.440 ms（FA4 の方が遅い）。
maxdoc=2048 の設定では FA4 が `cudaErrorIllegalAddress` でクラッシュ。→ **不採用**（FA4 の SM120 backward は SM80 世代 MMA ベースで smem 99 KB 制約、codex の予想通り）。
- 結果として採用した構成: micro-batch 4/8/12 (32768 tokens)、CE chunking はデフォルト off（`CE_CHUNK_ROWS` 環境変数で有効化可能）。
  - stage 3 を 8x49152 にすると chunked CE 込みで 1517 ms（−1%）だが reserved 26.8 GB + 非 torch ~4.4 GB でほぼ限界 → 不採用
  - 12x32768 と 16x24576 は同等（1532 vs 1529 ms）。harness のノイズは ±1.5% 程度
- bench: stage1 494 ms, stage2 1004 ms, stage3 1532 ms → 予測 train_time ≈ 1306 s
- 事前チェック: 12 step の損失 18.94 → 12.91（参照と一致、finite）

**結果**: `logs/f87cde40-8371-4e6a-bdad-25918515eb75.txt` — **train_time 1304.9 s, val_loss 3.2772**（Exp1 比 −8.7%、baseline 比 −23.4%）。途中 val_loss も一致（step250 4.4966, step1000 3.4167）。ピーク 21.5 GB。
（1 回目の起動は cwd ミスで `./run.sh` が見つからず GPU が 15 分遊んだ — 起動は絶対パスで）

#### codex による Exp 2b diff レビュー（2026-09-03）
- P1: `grad_scale = 1/grad_accum_steps` だと loss が token 和なので、stage ごとに勾配の大きさが 4x/2x/1.33x 変わる（旧 5090 port の 1/16 も H100x8 参照の実効 1/8 と 2x ずれていた）。Muon (polar express) / Adam / cautious WD はすべてスケール不変なので結果は変わらない（Exp 2b の loss も正常）が、stage 境界で Adam の 2-step 累積・momentum が一時的に歪む。→ 参照と完全に一致する定数 `grad_scale = world_size / 8` に変更予定（`ForwardScheduleConfig.grad_scale` の配管も不要になる）。
- P2: micro-batch 数を変えると境界（bigram hash の先頭、MTP 先読みの打ち切り）が変わる → データ境界の効果で、少ない方がむしろ良い。問題なし。
- 他（embedding 行 offset、tied embed/lm_head、odd-step Adam、CPU snapshot、CE chunk）は問題なしとの評価。

### Exp 3: FP8 MLP backward（2026-09-03）
Exp 2b 後のプロファイル: bf16 GEMM (cuBLAS) が step 時間の 37%（うち約半分が MLP backward の dW2/dW1/dx）、fp8 GEMM 16%、inductor elementwise 15%、fused MLP 13%、flash-attn 7-10%。
**変更** (`FusedLinearReLUSquareFunction.backward`, `MLP_FP8_BWD=1` がデフォルト):
- dW2 = post^T g, dpre_pre = g W2^T, dW1 = dpre^T x, dx = dpre W1 の 4 GEMM をすべて cuBLAS FP8 (`_scaled_mm`) に。gradient (g, dpre) は e5m2、activation (x, post) と weight は e4m3、per-tensor の動的スケール（lm_head backward と同じ流儀）
- W1^T / W2 の row-major fp8 コピーを `quantize_mlp_fp8` で step ごとに用意（cuBLAS FP8 は A row-major / B column-major のみ）
- 保存する activation は bf16 のまま（x, post）で backward 内で再量子化。forward の delayed-scaling スケール（in-place 更新される buffer）を backward で使うと、torch.compile の partitioner が backward 側で「更新後の buffer から clone を再計算」してしまい dW2 が 30x 狂う（compile でのみ再現）。
- fp32→fp8 の cast は compile 下では saturate しない（448 をわずかに超える値が NaN）→ cast 前に clamp
- e5m2 の範囲チェック: max 57344

**孤立テスト**: fp32 参照との rms 相対誤差 dx/dW1/dW2 = 9.1% / 9.1% / 7.3%（bf16 backward: 3.8% / 3.8% / 4.4%。どちらも forward の fp8 量子化誤差を含む）
**bench**: stage1 497→457 ms, stage2 1004→930 ms, stage3 1532→1421 ms（−7〜8%）。12 step 損失 18.94→12.91（参照と一致）。予測 train_time ≈ 1210 s。
**リスク**: e5m2 勾配による最終 loss の悪化。

**結果**: `logs/a8e283c2-215d-4e87-a39a-5d92422e9784.txt` — **train_time 1210.5 s, val_loss 3.2763**（Exp 2b 比 −7.2%、baseline 比 −29.0%）。loss は悪化せず（途中 val: step250 4.5300（他 run より +0.03 だが）, step750 3.6822, step1000 3.4146 と後半はむしろ良い）。ピーク 21.6 GB。
