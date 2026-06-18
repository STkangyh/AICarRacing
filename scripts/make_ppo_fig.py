"""Generate report_assets/fig_ppo_overview.png — a 2-panel PPO mechanism figure:
(left) clipped surrogate objective L^CLIP(r) for A>0 / A<0 with our ε=0.15;
(right) the PPO training loop (rollout -> GAE -> clipped minibatch update -> per-minibatch KL early-stop -> cosine LR -> repeat).
Matches the other report figures (Korean font, Agg)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

for _f in ("AppleGothic", "Apple SD Gothic Neo", "Nanum Gothic"):
    try:
        matplotlib.rcParams["font.family"] = _f; break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report_assets")
C = "#1f77b4"

fig = plt.figure(figsize=(11.5, 5.2))

# ---------- Panel A: clipped surrogate objective ----------
axA = fig.add_subplot(1, 2, 1)
eps = 0.15
r = np.linspace(0.6, 1.4, 400)
A_pos = np.minimum(r, 1 + eps)          # A>0: 상한 1+ε
A_neg = -np.maximum(r, 1 - eps)         # A<0: 하한 -(1-ε)
axA.axvspan(1 - eps, 1 + eps, color="green", alpha=0.07, label=f"신뢰영역 [1-ε, 1+ε], ε={eps}")
axA.plot(r, A_pos, color=C, lw=2.4, label="$A_t>0$ (좋은 행동): 1+ε에서 상한")
axA.plot(r, A_neg, color="#d62728", lw=2.4, label="$A_t<0$ (나쁜 행동): 1-ε에서 하한")
for xv in (1 - eps, 1.0, 1 + eps):
    axA.axvline(xv, color="gray", ls=":", lw=0.8)
axA.scatter([1 + eps, 1 - eps], [1 + eps, -(1 - eps)], color="black", s=18, zorder=5)
axA.annotate("상한(clip)", xy=(1 + eps, 1 + eps), xytext=(1.18, 0.78), fontsize=8,
             arrowprops=dict(arrowstyle="->", color="gray"))
axA.annotate("하한(clip)", xy=(1 - eps, -(1 - eps)), xytext=(0.62, -0.55), fontsize=8,
             arrowprops=dict(arrowstyle="->", color="gray"))
axA.set_xlabel("확률비  $r_t(\\theta)=\\pi_\\theta / \\pi_{old}$")
axA.set_ylabel("목적함수 기여  $L^{CLIP}$")
axA.set_title("PPO Clipped Surrogate Objective\n(신뢰영역 밖 갱신은 잘라내 collapse 억제)", fontsize=10)
axA.legend(fontsize=7.5, loc="upper left")
axA.grid(True, ls="--", lw=0.4, alpha=0.5)

# ---------- Panel B: training loop ----------
axB = fig.add_subplot(1, 2, 2)
axB.set_xlim(0, 10); axB.set_ylim(0, 10); axB.axis("off")
axB.set_title("PPO 학습 루프 (우리 구현)", fontsize=10)
steps = [
    ("① Rollout 수집\n64개 병렬 env → 32,768 step", "#e8f0fb"),
    ("② GAE Advantage 추정\nγ=0.99, λ=0.95", "#e8f0fb"),
    ("③ 미니배치 PPO 갱신 (K epoch × 2048)\nclip(ε=0.15) + value MSE + entropy(0.01)", "#fdebe8"),
    ("④ per-minibatch KL 검사\napprox_kl > 0.045 → early-stop", "#fdf3e0"),
    ("⑤ Cosine LR 갱신\n1e-4 → 1e-5", "#e8f0fb"),
]
ys = np.linspace(9.0, 1.0, len(steps))
boxw, boxh = 8.6, 1.15
for (txt, fill), y in zip(steps, ys):
    box = FancyBboxPatch((0.7, y - boxh / 2), boxw, boxh,
                         boxstyle="round,pad=0.06,rounding_size=0.12",
                         linewidth=1.1, edgecolor=C, facecolor=fill)
    axB.add_patch(box)
    axB.text(5.0, y, txt, ha="center", va="center", fontsize=8.2)
for i in range(len(ys) - 1):
    axB.add_patch(FancyArrowPatch((5.0, ys[i] - boxh / 2), (5.0, ys[i + 1] + boxh / 2),
                                  arrowstyle="-|>", mutation_scale=13, color="#444", lw=1.2))
# return loop arrow (bottom -> top)
axB.add_patch(FancyArrowPatch((0.7, ys[-1]), (0.7, ys[0]),
                              connectionstyle="arc3,rad=-0.45", arrowstyle="-|>",
                              mutation_scale=13, color="#888", lw=1.2))
axB.text(0.15, 5.0, "다음 rollout 반복", rotation=90, va="center", ha="center", fontsize=8, color="#666")

fig.tight_layout()
p = os.path.join(OUT, "fig_ppo_overview.png")
fig.savefig(p, dpi=130, bbox_inches="tight")
print("saved", p)
