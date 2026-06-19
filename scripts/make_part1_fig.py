"""Generate report_assets/fig_base_scaleup.png — Part I base 2-action scale-up curve.
Data = the documented milestones in Part I 3.1 (shaped reward):
  7.2M -> 643, 9.83M -> 837.99 (best ckpt, peak), 20.0M -> 689 (overshoot/decline).
Reference band = 3-action saved models' embedded shaped mean (637.95 / 674.55).
Message: peak ~10M then decline; ~10-12M is the sweet spot; 20M is overshoot.
Matches the other report figures (Korean font, Agg, ASCII minus)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _f in ("AppleGothic", "Apple SD Gothic Neo", "Nanum Gothic"):
    try:
        matplotlib.rcParams["font.family"] = _f; break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report_assets")
C = "#1f77b4"

steps = [7.2, 9.83, 20.0]          # global step (M)
rew   = [643, 837.99, 689]         # shaped reward
labels = ["7.2M\n643", "9.83M\n837.99 (best)", "20.0M\n689"]

fig, ax = plt.subplots(figsize=(8.2, 4.6))

# 3-action embedded-shaped reference band (637.95 ~ 674.55)
ax.axhspan(637.95, 674.55, color="gray", alpha=0.12)
ax.axhline(674.55, color="gray", ls="--", lw=0.9)
ax.text(20.2, 656, "3-action 저장 모델\n(임베디드 shaped 638~675, 참고)",
        fontsize=7.5, va="center", ha="right", color="#555")

# sweet-spot band ~10-12M
ax.axvspan(10, 12, color="green", alpha=0.07)
ax.text(11, 250, "sweet spot\n~10–12M", fontsize=8, ha="center", color="#2a7")

ax.plot(steps, rew, "-o", color=C, lw=2.2, ms=8, zorder=5)
for x, y, t in zip(steps, rew, labels):
    dy = 18 if y < 800 else -52
    ax.annotate(t, xy=(x, y), xytext=(x, y + dy), fontsize=8.5, ha="center",
                color="#222")

# peak marker
ax.scatter([9.83], [837.99], s=160, facecolors="none", edgecolors="#d62728", lw=2, zorder=6)
ax.annotate("peak", xy=(9.83, 837.99), xytext=(13.0, 820), fontsize=9, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728"))
# decline arrow
ax.annotate("", xy=(20.0, 689), xytext=(9.83, 837.99),
            arrowprops=dict(arrowstyle="->", color="#d62728", ls="--", lw=1.0, alpha=0.6))
ax.text(15.5, 745, "-149 (과학습)", fontsize=8, color="#d62728", rotation=-18)

ax.set_xlim(6, 21)
ax.set_ylim(200, 900)
ax.set_xlabel("학습 스텝 (global step, M)")
ax.set_ylabel("shaped reward (학습 신호)")
ax.set_title("Part I — 베이스 2-action 스케일업: 9.8M에서 정점 후 20M 퇴보", fontsize=11)
ax.grid(True, ls="--", lw=0.4, alpha=0.5)

fig.tight_layout()
p = os.path.join(OUT, "fig_base_scaleup.png")
fig.savefig(p, dpi=130, bbox_inches="tight")
print("saved", p)
