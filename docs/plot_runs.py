# val_loss vs train_time for every full run (logs/*.txt), labeled by experiment; adopted runs highlighted.
import re, glob, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = "/home/takuma/co/nanogpt-speedrun-rtx5090x1"
RUNS = {  # log id prefix -> (label, adopted?)
    "c0f0c03b": ("Baseline (H100x8 code as-is)", True),
    "9e0fc6e7": ("Exp1 DC attn bwd kernel", True),
    "f87cde40": ("Exp2b emb-row grads + 4/8/12 mb", True),
    "a8e283c2": ("Exp3 FP8 MLP backward", True),
    "f0446be1": ("Exp4 FP8 attn proj (rejected)", False),
    "51b47d7a": ("Exp5 stage3 batch16 (rejected)", False),
    "f54e7ab6": ("Exp6 FP8 attn fwd-only (rejected)", False),
    "7505e1b1": ("Exp7 const grad_scale", True),
    "3c506227": ("Exp8 sampled softmax (loss>3.28)", False),
    "7802c421": ("Exp9 sampled softmax + ext35", True),
    "f99740d6": ("Exp10 logQ x1.0 (rejected)", False),
    "0535c2db": ("Exp11 tail averaging (rejected)", False),
    "5984bf8e": ("Exp12 SNS_START=0", True),
    "345def6d": ("Exp13 logQ x0.5 (RECORD)", True),
    "07c71c90": ("Exp14 repeat of Exp13 (RECORD)", True),
    "ec707ab7": ("Exp15 trims (not reproducible)", False),
    "fef82d5b": ("Exp16 10 layers (rejected)", False),
    "fa1b8ce0": ("Exp17 10 layers +50 steps (rejected)", False),
    "81497046": ("Exp18 c_proj init + TailEMA (rejected)", False),
    "c9aedb76": ("Exp19 grouped MUDD (rejected)", False),
    "ea261356": ("Exp20 stage3 doc cap 3072 (rejected)", False),
    "bdc85e88": ("Exp21 repeat of Exp15 (>3.28)", False),
    "35d44dad": ("Exp22 top-2048 always in candidates (rejected)", False),
}
pat = re.compile(r"^step:(\d+)/(\d+) val_loss:([\d.na]+) train_time:(\d+)ms")
runs = []
for f in sorted(glob.glob(f"{REPO}/logs/*.txt"), key=os.path.getmtime):
    rid = os.path.basename(f)[:8]
    if rid not in RUNS:
        continue
    pts = []
    for line in open(f, errors="ignore"):
        m = pat.match(line)
        if m:
            loss = m.group(3)
            if loss == "nan": break
            pts.append((int(m.group(4)) / 1000.0, float(loss), int(m.group(1)), int(m.group(2))))
    if len(pts) < 3 or pts[-1][2] != pts[-1][3]:
        continue  # incomplete run
    label, adopted = RUNS[rid]
    runs.append((rid, label, adopted, pts))
fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1.15, 1]})
cmap = plt.get_cmap("tab20")
def expno(label):
    m = re.match(r"Exp(\d+)", label)
    return m.group(1) if m else "B"
for k, (rid, label, adopted, pts) in enumerate(runs):
    t = [p[0] for p in pts]; l = [p[1] for p in pts]
    ls = "-" if adopted else "--" if adopted is False else "-."
    lw = 2.2 if adopted else 1.1
    axes[0].plot(t, l, ls, lw=lw, color=cmap(k % 20), label=f"{expno(label)}: {label} [{rid}]")
axes[0].set_xlabel("train_time [s]"); axes[0].set_ylabel("val_loss"); axes[0].set_title("val_loss vs train_time, all full runs (solid = adopted, dashed = rejected)")
axes[0].set_ylim(3.2, 4.8); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=6.6, loc="upper right")
# right panel: final points, numbered; adopted progression connected
prog = [(pts[-1][0], pts[-1][1]) for rid, label, adopted, pts in runs if adopted]
axes[1].plot([q[0] for q in prog], [q[1] for q in prog], color="gray", lw=1, alpha=0.6, zorder=1, label="adopted progression (baseline -> record)")
for k, (rid, label, adopted, pts) in enumerate(runs):
    tf, lf = pts[-1][0], pts[-1][1]
    axes[1].scatter([tf], [lf], color=cmap(k % 20), s=70 if adopted else 45, marker="o" if adopted else "x", zorder=5,
                    label=f"{expno(label)}: {label}  ({tf:.0f} s, {lf:.4f})")
    axes[1].annotate(expno(label), (tf, lf), fontsize=8, fontweight="bold", xytext=(4, 3), textcoords="offset points")
axes[1].axhline(3.28, color="red", lw=1, ls=":", label="target val_loss 3.28")
axes[1].set_xlabel("train_time [s]"); axes[1].set_ylabel("final val_loss"); axes[1].set_title("final val_loss vs train_time (numbers = experiment)")
axes[1].set_xlim(950, 1750); axes[1].set_ylim(3.272, 3.294); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=6.4, loc="upper right", ncol=1)
plt.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/docs/train_time_vs_val_loss.png"
plt.savefig(out, dpi=130)
print("saved", out, "runs:", len(runs))
for rid, label, adopted, pts in runs:
    print(f"  {rid} {pts[-1][0]:8.1f}s {pts[-1][1]:.4f} {label}")
