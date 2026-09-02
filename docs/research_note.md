# autoresearch: NanoGPT speedrun on RTX 5090 x1 — 結果ノート

目的: `val_loss <= 3.28` 到達までの `train_time` を短縮する（1x RTX 5090, 32GB, sm_120）。

## 現在の最小 train_time

| 日付 | train_time | val_loss | log | 変更 |
|---|---|---|---|---|
| 2026-09-02 | 1703.7 s | 3.2805 | `logs/c0f0c03b-9c5b-489a-a163-c4fff78b2518.txt` | baseline (H100x8 コードをそのまま RTX5090x1 で実行) |
| 2026-09-02 | **1429.8 s** | 3.2779 | `logs/9e0fc6e7-a0de-4a99-a7dc-454fcea70fd6.txt` | Exp 1: DC attention backward カーネル書き直し (−16.1%) |

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
