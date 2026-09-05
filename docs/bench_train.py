"""Short stage benchmark of the current training code; never a speedrun result.

Run from the repository root:
    uv run torchrun --standalone --nproc_per_node=1 docs/bench_train.py
    BENCH_STAGES=0,847 BENCH_PROFILE=nsys nsys profile \
        --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
        --capture-range=cudaProfilerApi --capture-range-end=repeat \
        -o logs/stages uv run torchrun --standalone --nproc_per_node=1 docs/bench_train.py

The real model, loader, optimizer and schedule are used. Warmup changes weights,
so only finite losses and timings are meaningful; validation is not evaluated.
"""

import contextlib
import os
from pathlib import Path
import statistics
import sys
import time

import torch


repo = Path(__file__).resolve().parents[1]
os.chdir(repo)
sys.path.insert(0, str(repo))
train_path = repo / "train_gpt.py"
source = train_path.read_text()
marker = "########################################\n#            Warmup kernels"
assert source.count(marker) == 1
# Reuse the complete setup, stopping before the speedrun's warmup/training loops.
sys.argv[0] = str(train_path)
torch.manual_seed(42)
ns = {"__name__": "__main__", "__file__": str(train_path)}
exec(compile(source.split(marker)[0], str(train_path), "exec"), ns)
model = ns["model"]
manager = ns["training_manager"]
print0 = ns["print0"]
print0("BENCHMARK ONLY: not a full training run or record", console=True)
prefix = ns["build_prefix_table"](model.vocab_size)
model.prefix_table.copy_(prefix)
ns["sns_builder"].prefix_table = prefix.numpy()
loader = ns["distributed_data_generator"](
    ns["args"].train_files,
    ns["TRAINING_STAGES"][0].batch_size,
    ns["TRAINING_STAGES"][0].train_max_seq_len,
    grad_accum_steps=manager.grad_accum_steps,
)
model.train()
profile_mode = os.environ.get("BENCH_PROFILE", "")
stages = [int(v) for v in os.environ.get("BENCH_STAGES", "0,423,847,1129,1205,1270").split(",")]
warmup = int(os.environ.get("BENCH_WARMUP", "10"))
repeats = int(os.environ.get("BENCH_STEPS", "6"))
assert warmup >= 2 and repeats >= 2
print0(Path(__file__).read_text())
print0(f"BENCH config: stages={stages} warmup={warmup} repeats={repeats} profile={profile_mode!r} seed=42", console=True)


def run_step(step):
    manager.advance_schedule(step)
    loss_sum = torch.zeros((), device="cuda")
    for _ in range(manager.grad_accum_steps):
        with torch.cuda.nvtx.range("load_and_candidates"):
            inputs, targets, lengths, bigrams, bigram_cpu, target_cpu = loader.send(manager.train_loader_send_args)
            manager.sparse_index_update(step, bigram_cpu)
            candidates = ns["sns_args"](step, target_cpu)
        with torch.cuda.nvtx.range("gather_rows"):
            rows = ns["gather_embedding_rows"](model, inputs, bigrams)
        with torch.cuda.nvtx.range("forward_backward"):
            loss = model(inputs, targets, lengths, bigrams, manager.get_forward_args(), *rows, *candidates).sum() * ns["grad_scale"]
            manager.sparse_index_share(step)
            loss.backward()
            loss_sum.add_(loss.detach())
        with torch.cuda.nvtx.range("accumulate_rows"):
            ns["accumulate_embedding_grads"](model, inputs, bigrams, rows)
        del loss, rows
    with torch.cuda.nvtx.range("optimizer"):
        manager.step_optimizers(step)
        model.quantize_mlp_fp8(bootstrap_down=(step < 16))
    return loss_sum / (manager.batch_size * ns["grad_scale"])


for stage in stages:
    # Repeat the first even/odd pair to preserve the schedule and Adam cadence.
    # At odd boundaries, one setup step applies the exact transition first.
    if stage % 2:
        run_step(stage)
    pair_start = stage + stage % 2
    for i in range(warmup):
        loss = run_step(pair_start + i % 2)
        value = loss.item()
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Nonfinite loss at stage {stage}, warmup {i}: {value}")
        if stage == 0:
            print0(f"SANITY step={i} loss={value:.6f}", console=True)
    torch.cuda.synchronize()
    if profile_mode == "nsys":
        torch.cuda.cudart().cudaProfilerStart()
    profile_context = (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        if profile_mode == "torch" else contextlib.nullcontext()
    )
    durations = []
    with profile_context as profiler:
        for i in range(repeats):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.cuda.nvtx.range(f"stage_{stage}_step_{i}"):
                loss = run_step(pair_start + i % 2)
            torch.cuda.synchronize()
            durations.append(1000 * (time.perf_counter() - start))
    if profile_mode == "nsys":
        torch.cuda.cudart().cudaProfilerStop()
    if profile_mode == "torch":
        profiler.export_chrome_trace(str(repo / "logs" / f"bench_stage_{stage}.json"))
        print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
    print0(
        f"BENCH stage={stage} candidates={ns['sns_p_at'](pair_start)} "
        f"microbatches={manager.grad_accum_steps} mean_ms={statistics.mean(durations):.3f} "
        f"median_ms={statistics.median(durations):.3f} loss={loss.item():.6f} "
        f"peak_MiB={torch.cuda.max_memory_allocated() / 1024**2:.1f}",
        console=True,
    )
ns["dist"].destroy_process_group()
