# autoresearch: NanoGPT speedrun on RTX 5090 x1 — 結果ノート

目的: `val_loss <= 3.28` 到達までの `train_time` を短縮する（1x RTX 5090, 32GB, sm_120）。

## 現在の最小 train_time

| 日付 | train_time | val_loss | log | 変更 |
|---|---|---|---|---|
| 2026-09-02 | 1703.7 s | 3.2805 | `logs/c0f0c03b-9c5b-489a-a163-c4fff78b2518.txt` | baseline (H100x8 コードをそのまま RTX5090x1 で実行) |
| 2026-09-02 | 1429.8 s | 3.2779 | `logs/9e0fc6e7-a0de-4a99-a7dc-454fcea70fd6.txt` | Exp 1: DC attention backward カーネル書き直し (−16.1%) |
| 2026-09-03 | 1304.9 s | 3.2772 | `logs/f87cde40-8371-4e6a-bdad-25918515eb75.txt` | Exp 2b: embedding 行勾配 + micro-batch 4/8/12 + MLP カーネル設定 (−8.7%) |
| 2026-09-03 | 1210.5 s | 3.2763 | `logs/a8e283c2-215d-4e87-a39a-5d92422e9784.txt` | Exp 3: FP8 MLP backward (−7.2%) |
| 2026-09-03 | 1093.3 s | 3.2789 | `logs/7802c421-1dca-47c1-816e-79e33fb0d9b2.txt` | Exp 9: sampled softmax 訓練損失（PR #360 移植）+ 拡張 35 step (−9.8%) |
| 2026-09-03 | 1086.4 s | 3.2805 | `logs/5984bf8e-aa01-435e-b220-a93266662c89.txt` | Exp 12: Exp 9 の SNS_START=0（序盤 full softmax 無し）。val は 3.28 ちょうど |
| 2026-09-03 | **1093.1 s** | 3.2766 | `logs/345def6d-2a4c-4067-be97-4e8658b42cf4.txt` | Exp 13: 部分的 log-Q 補正 (×0.5)。val にマージンが戻る（現在の記録構成） |

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

### 調査: torch 2.11.0+cu130 (cuDNN 9.19, triton 3.6)（2026-09-03）
別 venv で同じコードを bench: stage1 451 ms / stage2 915 ms / stage3 1486 ms（2.10+cu128: 452 / 920 / 1409）。fwd+bwd は同等、optimizer step が 25→113 ms に悪化（stage3 のみ）。→ **見送り**。
なお CE カーネルの `_compile_kernel(compute_capability="90")` は cu130 では動かない（PTX JIT 不可）ので実 GPU の SM 向けにコンパイルするよう変更済み。DC カーネルの loop-carried 変数も新しい Triton では fp64 に昇格してコンパイル失敗するため明示 cast を追加。

### Exp 4: FP8 attention projections + 定数 grad_scale（2026-09-03）
- attention の QKV / O projection (fwd+bwd) を FP8 化（`FP8LinearFunction`、MLP backward と同じ流儀: e4m3 activation/weight, e5m2 grad, 動的 per-tensor scale, cast 前 clamp）。eval では bf16 のまま（val の計算は不変）。
- `grad_scale = world_size / 8`（codex レビュー P1: 参照 8xH100 の正規化と一致させ、stage ごとに変わらないように）。`ForwardScheduleConfig.grad_scale` の配管を撤去。
- プロファイル: bf16 GEMM はほぼ消え fp8 GEMM 587 ms/step（~450 TFLOPS、ほぼ roofline）。ただし fp8 の量子化/転置/amax パス（inductor 生成）が ~220 ms/step（stage3 の 16%）あり、attention FP8 の正味は −2%（452→448 / 920→910 / 1409→1394 ms）。
- 試作: 「量子化 + row-major/転置 同時出力」の fused Triton カーネル（`scratchpad/prof/fp8_quant2d.py`）— inductor が既に同等の融合をしていたため効果なし（1%）。tl.trans + 2 出力の組合せで miscompile する罠あり。簡潔さ優先で不採用（delayed scaling で amax パスを省けば +3% 程度の見込みだが状態管理が増えるので保留）。
- 見つけた torch.compile の罠: forward で in-place 更新される buffer から作った clone/slice を backward 用に保存すると、partitioner が backward 側で「更新後の buffer から再計算」してしまう。compile 内での `_used.copy_(buf)` も functionalization で alias になるので、スナップショットは compile の外で取る必要がある。
- 12 step 損失 18.94→12.90（参照一致）。bench: 448 / 910 / 1394 ms（予測 ≈ 1190 s）。

**結果 (不採用)**: `logs/f0446be1-7567-49d2-a049-aba22987d08f.txt` — train_time 1194.9 s だが **val_loss 3.2837**（Exp 3 の 3.2763 から +0.0074、目標 3.28 を超過）。step1250 でも 3.3002 vs 3.2928 と一貫して悪い。attention 射影の e5m2 勾配（QK/VO bank は Muon）が効いていると判断。時間 −1.3% に対し loss 悪化は ~30 step 分（+3%）→ **attention FP8 は却下**、コードは Exp 3 に戻す。
（定数 grad_scale は理論上スケール不変で無害のはずだが、この run に同居させたため単独での検証はできていない。Exp 3 の per-stage 版を維持。）

### 計画: Exp 5 — stage 3 のグローバルバッチを 24→16 (x2048x8) に（single-GPU 向けスケジュール実験）
H100x8 では通信/MFU の都合で大バッチが有利だったが、1 GPU では micro-batch サイズ (32768 tokens) が変わらないため、stage 3 の
バッチを小さくして optimizer step を増やしても step あたりのコストはほぼ比例（overhead ~1.5%）。token 効率が上がれば勝ち。
- 実装 (`STAGE3_BATCH_UNITS=16`): stage 3 を 634 step + 拡張 22 step（token 数 338.56M ≈ 参照 338.82M）、lr_mul = 1.73·sqrt(16/24) = 1.41
- LR の cooldown と Muon momentum cooldown は「累積 token」で参照スケジュール（バッチ 24）に写像（CPU で検証: lr-decay の差 1e-16、momentum 差 0.002）
- 判定: 同 token で val_loss が有意に下がれば（≦3.270 程度）、token 数を削って train_time を縮める Exp 6 へ
- bench: stage3 976 ms/step (8 micro-batch) × 634 step ≈ 619 s vs 1421 ms × 424 ≈ 602 s（同 token で +2.8%、optimizer step の固定費 ~25-45 ms が効く）

- 1 回目の起動 (`logs/7964745c-...`) は拡張 stage の lr_mul が 1.41 のまま（参照は 1.0 → 最後の 22 step の LR が 0.212 vs 0.15）だったので 5 分で中断して修正・再起動。

**結果 (不採用)**: `logs/51b47d7a-5be7-4bb0-bc2e-71de254c9a48.txt` — **val_loss 3.2772, train_time 1213.3 s**（Exp 3: 3.2763 / 1210.5 s）。同 token で loss は同じ（ノイズ範囲）、時間は +0.2%。stage 3 のバッチを小さくしても token 効率は上がらない（この loss 帯では 393k tokens のバッチはまだ critical batch size 以下ということ）→ **却下**、スケジュールコードは戻す（`scratchpad/prof/patch_stage3_batch.py` に保存）。
途中経過: step1000 3.4524（参照は 3.4146 だが token 位置が違う: 1000 step = 206.7M tokens vs 参照の 1000 step = 231M）。

### 調査: FlexAttention vs flash-attn2（2026-09-03）
varlen + sliding window + causal の block mask で比較（T=16k-32k, window 128-1408）: forward は同等、backward は Flex が ~50% 遅い（例 T=32768 W=896: FA2 1.65 ms vs Flex 2.40 ms fwd+bwd）。→ FA2 のまま。attention カーネルの選択肢は FA2 / FA4 / Flex を試して FA2 が最良。

### 試作: e5m2 勾配の delayed scaling + fused 量子化カーネル（2026-09-03、不採用）
amax パスを省くため、MLP backward の g / dpre の per-tensor scale を「前 micro-batch の amax × 2」に（buffer を backward 内で更新: torch.compile でも動作することを確認）。数値は動的 scale 版と同等（bf16 backward 比 rms 誤差 6-8%）だが、**速度は変わらず**（456 / 926 / 1417 ms vs 457 / 930 / 1421）。inductor は既に amax reduction を rms_norm backward 等に融合しており、残る FP8 overhead は K-major 転置の書き出しなど本質的なもの。→ 却下（`scratchpad/prof/patch_delayed_scaling.py`）。

### 調査: inductor のチューニングオプション（2026-09-03）
`TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1`, `TORCHINDUCTOR_MULTI_KERNEL=1`: fwd/bwd の時間は不変（428 / 1348 ms）。→ 不採用。（coordinate descent tuning は upstream で禁止されているので試していない）

### 調査: lm_head/CE の行チャンク化で logits を L2 常駐に（2026-09-03、不採用）
`CE_CHUNK_ROWS` = 1024 / 2048 / 4096: stage3 1482 / 1462 / 1435 ms（無チャンク 1421 ms）。小さい GEMM とループのオーバーヘッドが L2 効果を上回る。

### Exp 6: attention 射影の forward のみ FP8（2026-09-03）
Exp 4 の loss 悪化は e5m2 勾配が原因と推定し、QKV / O projection の **forward のみ** e4m3 (動的 per-tensor scale) にして backward は bf16 のまま（`FP8LinearFwdFunction`, `ATTN_FP8_FWD=1`）。MLP up-proj の forward FP8 は upstream でも採用済みなのでリスクは低いはず。
bench: stage1 457→446 ms, stage3 1421→1388 ms（−2.3%）。12 step 損失 18.94→12.94（参照範囲）。

**結果 (不採用)**: `logs/f54e7ab6-2fd0-4b00-830d-4803d9125aac.txt` — train_time 1183.1 s (−2.3%) だが **val_loss 3.2808**（Exp 3 の 3.2763 から +0.0045、step1250 でも +0.004 と一貫）。forward の e4m3 量子化だけでも attention 射影は loss に効く（MLP と違って v / attention 出力の誤差が直接残差に入るため?）。拡張 stage を ~10 step 増やせば相殺できるが正味 ~1% にしかならず、コードも増えるので **却下**。Exp 3 が引き続き最良。

### Exp 7: 定数 grad_scale = world_size/8 の検証（= Exp 3 構成の再実行）（2026-09-03）
codex レビュー P1 の対応（参照 8xH100 の勾配正規化と一致させ、`ForwardScheduleConfig.grad_scale` の配管を撤去）。optimizer は全てスケール不変なので結果は Exp 3 と同じはず。同時に、同一構成の再実行として run 間ばらつきの目安を得る。

**結果 (採用)**: `logs/7505e1b1-711f-4a1d-ac12-edd500db412f.txt` — **val_loss 3.2782, train_time 1212.1 s**（Exp 3: 3.2763 / 1210.5 s）。差はノイズ範囲（同構成 4 run: 3.2763 / 3.2772 / 3.2772 / 3.2782、sd ≈ 0.0008 → 有意な差は |Δ| > ~0.002）。定数 grad_scale を採用（コードが簡潔、参照と同じ正規化）。

### 同一構成の run 間ばらつき（参考）
Exp 2b/3/5/7 系（最終 val_loss）: 3.2772, 3.2763, 3.2772, 3.2782 → 平均 3.2772, sd 0.0008。train_time は ±0.2%。

### Exp 8: Sampled softmax 訓練損失（upstream PR #360 の移植）（2026-09-03）
upstream の未マージ PR #360 (@devenpzak, "ANVIL2", 8xH100 で 73.9→39.9 s) のアブレーションで、single GPU に移植可能かつ効果が大きいのが
「shared-negative sampled softmax（訓練損失のみ、validation は完全な 50304-way softmax のまま）」（H100 で −8.8 s wall, val −2.4 millinats）。
（最大の要因である 84.6M 行の hashed n-gram table（65B params、8 GPU に shard）は 32 GB では不可能。mixed-width attention は patched FA3 (sm_90) 依存で不可。）

**実装** (`SampledSoftcappedCrossEntropy`, `sns_p_at`, `SampledSoftmaxCandidates`):
- micro-batch ごとにホスト側で候補集合 C を作る: その micro-batch の全 target ＋ stride 順列 (k·20011 mod V) からの負例、昇順で P 個。target/prefix の C 内位置 (TPOS/PPOS) を渡す（prefix が C に無ければ −1 = 無視）
- lm_head の fp8 重みから C の行を gather し、logits GEMM / CE カーネル（VOCAB_SIZE=P でコンパイル）/ backward の 2 GEMM を P 幅で実行。重み勾配は (768, V) に zero 埋めで densify
- P: stage 1-2 = 10240（32k-token micro-batch の unique target は最大 ~7.6k → 34% headroom）、stage 3 = 14336 → 14336 → 24576 のランプ、最後の 100 step は完全 softmax。P の切り替え step は warmup に含める（グラフ再コンパイル回避）
- 正しさ: C = 全語彙にすると full softmax と loss/勾配が bit-exact。P=10240 で dx の相対差 1.5%
- コスト: T=8192 の lm_head+CE fwd+bwd 6.5 → 2.1 ms

**bench**: stage1 457→390 ms, stage2 930→793 ms, stage3 (P=14336) 1421→1230 ms（−13〜15%）。予測 train_time ≈ 1060 s。
**リスク**: 訓練勾配のバイアス → 最終 loss。PR の証拠では val はわずかに改善。

**結果 (loss 未達)**: `logs/3c506227-5f6c-4a28-ad45-03fffcb0ab4d.txt` — **train_time 1060.8 s (−12.5%) だが val_loss 3.2862**（+0.009）。途中の val が大きく悪い（step250 4.76 vs 4.50, step750 3.84 vs 3.69, step1000 3.50 vs 3.42）→ sampled softmax は序盤の学習を大きく遅らせ、最後の 100 step の full softmax で大半を取り戻すが 0.009 残る。PR #360 のアブレーション表を読み直すと、彼らの環境でも sampled softmax は val を 2.4 millinats 悪化させており（wall −8.8 s との交換）、私の環境では 1 GPU あたり micro-batch が 32k tokens（unique target ~7k）なので負例比率が低く、バイアスがより大きい可能性。
最終 0.009 は終盤の 1 step ≈ 0.0005 換算で ~20 step (~28 s) 相当 → 正味ではまだ −9% 程度の余地。

### Exp 9: Exp 8 + 序盤 100 step は full softmax + 拡張 step 15→35（2026-09-03）
- `SNS_START=100`: step < 100 は full softmax（序盤の統計学習を邪魔しない; コスト +7 s）
- `NUM_EXTENSION_ITERATIONS=35`: 最終 LR での拡張 step を 20 増やして loss 差 0.009 を回収（+28 s）
- 予測: train_time ≈ 1095 s、val_loss ≈ 3.277-3.280

**結果 (採用)**: `logs/7802c421-1dca-47c1-816e-79e33fb0d9b2.txt` — **train_time 1093.3 s, val_loss 3.2789**（−9.8%、baseline 比 −35.8%）。3.28 は下回るがマージンは薄い（同構成ノイズ ±0.001）。途中 val は依然として悪い（step250 4.73, step1000 3.51 vs 3.42）→ 序盤 100 step の full softmax は効かず、sampled 期間中のバイアスが主因。`NUM_EXTENSION_ITERATIONS` のデフォルトを 35 に変更してコミット。

#### codex による Exp 8 diff レビュー
- 位置ベースの MTP 先読み・prefix 処理・grad_scale・P ごとの warmup・validation 経路に問題なし。
- 指摘 1: stride 順列のスライスが末尾で wrap しない（負例の一部をスキップ；P は while ループで埋まるので結果は正しい）→ 順列を 2 倍長にして修正（次 run から）。
- 指摘 2: warmup が `sns_builder.off` を進めるのでリセット後の開始オフセットが warmup 依存 → reset で 0 に戻す（次 run から）。
- 指摘 3（既存の挙動）: lm_head.grad を fp32 で返しても param が bf16 なので autograd が bf16 に cast して累積している（upstream からそのまま）。

### Exp 10: sampled softmax の log-Q 補正（2026-09-03）
Exp 8/9 の sampled 期間中の val 悪化は、候補外クラスの logit に勾配が流れない系統的バイアスによる。古典的な sampled-softmax の補正: 負例（一様サンプル、採用確率 (P−U)/(V−U)）の exp(logit) を (V−U)/(P−U) 倍する = softcap 後の logit に log((V−U)/(P−U)) を加算（target クラスは常に候補なので 0）。CE カーネルに per-class offset ベクトルを追加（full softmax ではゼロ）。
- 孤立テスト: 損失の期待値が full と一致（19.60 vs 19.78; 補正なし 16.87）。16 回の負例サンプルで平均した dx の相対誤差 0.027(1 回)→0.0069（→0 に収束 = 不偏）、補正なしは 0.012 で頭打ち（バイアス）。代償として 1 micro-batch あたりのノイズは増える（負例列の重み勾配が ×~14）。
- 構成は Exp 9 と同一（SNS_START=100, 拡張 35 step）で補正の効果だけを見る。

**結果 (不採用)**: `logs/f99740d6-1f22-4e57-89dc-54d303210ec0.txt` — train_time 1099.3 s、**val_loss 3.2846**（Exp 9 の 3.2789 より悪い）。途中 val はバイアス除去で大幅改善（step250 4.517 vs 4.727、step1000 3.447 vs 3.506、full softmax 参照 4.50 / 3.42 に近い）が、step1250 以降で逆転（3.3067 vs 3.3011）。負例の ×(V−U)/(P−U) 重みによる勾配ノイズ（特に lm_head 列）が終盤の精度に効いたと解釈。→ 補正は却下（コードは Exp 9 に戻す）。中庸（部分補正、大きい P）は今後の候補。

### Exp 11: Exp 9 + 終盤 300 step の tail averaging（2026-09-03）
PR #360 / #347 の "ships" の簡略版: 最後の 300 step の重みの fp32 EMA（lm_head/embed は Adam step ごと、bank/value_embeds は 4 step ごと）を、最終 validation 直前（クロック内）に重みへ blend（lm_head/embed 0.65、bank と value_embeds 0.5）。codex 指摘の 2 点（順列 wrap、warmup 後のオフセット reset）も反映。
期待: 終盤の EMA で val −0.002〜0.005 → 拡張 step を減らす余地。

- 1 回目 (`logs/5771bcf3-...`) は step 1005（tail 開始）で `KeyError: 'lm_head'`（compiled model の named_parameters は `_orig_mod.` 接頭辞付き; CPU テストは素のモジュールだった）→ param の `.label` で引くよう修正して再実行。

**結果 (不採用)**: `logs/0535c2db-e6e8-43eb-a93a-c094ba017f6f.txt` — train_time 1093.8 s、**val_loss 3.2811**（Exp 9 の 3.2789 より +0.002）。step1250 までは Exp 9 と同じ軌跡（3.3007 vs 3.3011）なので ship 自体がわずかに悪化させた（LR 0.15 でまだ改善中の bank を 200 step 遅れの EMA と 0.5 で混ぜるのは早すぎる/重すぎる）。→ 却下（`scratchpad/prof/patch_tail_ships.py`）。PR #347 流の lm_head/embed のみ TailEMA（blend 0.65）は未検証。

### Exp 12: Exp 9 構成（+ codex 修正）で SNS_START=0（2026-09-03）
Exp 8→9 で序盤 100 step の full softmax は step 250 の val をほぼ改善しなかった（4.76→4.73）ので外す（−7 s）。同時に、記録構成の再現性（3.28 以下に安定して入るか、sampled 期の run 間ノイズ）を測る。

**結果 (採用、ただしマージン無し)**: `logs/5984bf8e-aa01-435e-b220-a93266662c89.txt` — **train_time 1086.4 s, val_loss 3.2805**。軌跡は Exp 9 と同じ（step250 4.70 vs 4.73、step1250 3.3022 vs 3.3011）→ 序盤の full softmax は不要（−7 s）。最終 val は 3.2789 / 3.2805 の 2 run で、sampled 構成のノイズは full 構成と同程度（±0.001）だが 3.28 ぎりぎり。baseline 自体が 3.2805 だったので「3.28 程度」の範囲だが、次はバイアス低減でマージンを回復したい。

### Exp 13: 部分的 log-Q 補正（SNS_LOGQ_SCALE=0.5）（2026-09-03）
完全補正（Exp 10）は途中 val を大きく改善したが負例の重み ×14 のノイズで最終 val が悪化。オフセットを半分（log 比 × 0.5 → 重み ×~3.7）にしてバイアスとノイズの中間を取る。構成は Exp 12 と同じ（SNS_START=0, 拡張 35）。

**結果 (採用)**: `logs/345def6d-2a4c-4067-be97-4e8658b42cf4.txt` — **train_time 1093.1 s, val_loss 3.2766**。途中 val も大幅改善（step250 4.54 vs 4.70、step1000 3.446 vs 3.498、step1250 3.2980 vs 3.3022）で full softmax 系（平均 3.2772）と同水準に戻った。完全補正 (×1.0) の 3.2846 より良く、バイアスとノイズの中間が正解。時間は Exp 12 比 +7 s（offset ベクトルのホスト計算/転送 or ノイズ）。
拡張 step の削減余地: 20 step ≈ 0.007（Exp 8→9）なので 10 step 削ると 3.280 前後になり不可。デフォルト `SNS_LOGQ_SCALE=0.5` でコミット。

### Exp 14: Exp 13 構成の再実行（run 間ばらつき）（2026-09-03）
記録構成（sampled softmax ×0.5 補正、拡張 35）が安定して 3.28 を下回るかの確認。

**結果**: `logs/07c71c90-dbb3-4316-86c5-a8ae01260f6c.txt` — **val_loss 3.2777, train_time 1093.6 s**。2 run で 3.2766 / 3.2777（平均 3.2772、full softmax 系と同じ分布）→ 記録構成は再現性あり、3.28 に対し ~0.003 のマージン。

### Exp 15: マージンを時間に換える（2026-09-03）
Exp 13/14 の余裕 ~0.003 を使う 3 つの小さな削減を同時に: stage 3 の候補ランプ無し（P=14336 固定; 最後の 1/3 で 24576 → −7 s）、終盤の full softmax を 100→60 step（−6 s）、拡張 step 35→30（−7 s）。合計 −20 s の見込み。3.28 を超えたら全て戻す。

**結果**: (実行中)

