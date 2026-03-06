
import matplotlib.pyplot as plt
import numpy as np

results = {
    "RNN": {
        "epochs" : list(range(1, 51)),
        "loss"   : [5.70, 5.63, 5.55, 5.55, 5.55, 5.55, 5.55, 5.45, 5.43, 5.43,
                    5.42, 5.42, 5.41, 5.41, 5.40, 5.40, 5.40, 5.39, 5.39, 5.39,
                    5.39, 5.38, 5.38, 5.38, 5.38, 5.37, 5.37, 5.37, 5.37, 5.36,
                    5.36, 5.36, 5.36, 5.35, 5.35, 5.35, 5.35, 5.34, 5.34, 5.34,
                    5.34, 5.33, 5.33, 5.33, 5.33, 5.32, 5.32, 5.32, 5.32, 5.31],
        "acc"    : [0.90, 1.10, 1.16, 1.16, 1.18, 1.18, 1.17, 1.28, 1.31, 1.31,
                    1.32, 1.33, 1.34, 1.34, 1.35, 1.35, 1.36, 1.36, 1.37, 1.37,
                    1.38, 1.38, 1.39, 1.39, 1.40, 1.40, 1.41, 1.41, 1.42, 1.42,
                    1.43, 1.43, 1.44, 1.44, 1.45, 1.45, 1.46, 1.46, 1.47, 1.47,
                    1.48, 1.48, 1.49, 1.49, 1.50, 1.50, 1.51, 1.51, 1.52, 1.52],
        "color"  : "#e74c3c",
    },
    "GRU": {
        "epochs" : list(range(1, 51)),
        "loss"   : [4.13, 3.83, 3.79, 3.79, 3.81, 3.85, 3.90, 3.93, 3.95, 3.97,
                    3.99, 4.01, 4.02, 4.03, 4.04, 4.05, 4.06, 4.07, 4.07, 4.08,
                    4.09, 4.09, 4.10, 4.10, 4.11, 4.11, 4.12, 4.12, 4.13, 4.13,
                    4.13, 4.14, 4.14, 4.14, 4.15, 4.15, 4.15, 4.16, 4.16, 4.16,
                    4.17, 4.17, 4.17, 4.17, 4.18, 4.18, 4.18, 4.18, 4.19, 4.19],
        "acc"    : [11.83, 15.33, 15.82, 15.72, 15.29, 14.58, 13.91, 13.40, 13.00, 12.70,
                    12.45, 12.25, 12.08, 11.94, 11.82, 11.72, 11.63, 11.55, 11.48, 11.42,
                    11.37, 11.32, 11.28, 11.24, 11.20, 11.17, 11.14, 11.11, 11.08, 11.06,
                    11.04, 11.02, 11.00, 10.98, 10.96, 10.95, 10.93, 10.92, 10.90, 10.89,
                    10.88, 10.87, 10.86, 10.85, 10.84, 10.83, 10.82, 10.81, 10.80, 10.79],
        "color"  : "#f39c12",
    },
    "Transformer": {
        "epochs" : list(range(1, 51)),
        "loss"   : [4.51, 4.18, 4.09, 4.04, 4.01, 3.97, 3.95, 3.93, 3.91, 3.89,
                    3.87, 3.85, 3.83, 3.81, 3.79, 3.77, 3.75, 3.74, 3.72, 3.71,
                    3.69, 3.68, 3.67, 3.65, 3.64, 3.63, 3.62, 3.61, 3.60, 3.59,
                    3.58, 3.57, 3.56, 3.55, 3.54, 3.53, 3.52, 3.51, 3.50, 3.49,
                    3.48, 3.47, 3.47, 3.46, 3.45, 3.44, 3.44, 3.43, 3.42, 3.42],
        "acc"    : [7.12, 10.12, 11.02, 11.62, 11.99, 12.48, 12.76, 13.08, 13.32, 13.60,
                    13.85, 14.08, 14.29, 14.48, 14.66, 14.82, 14.97, 15.11, 15.24, 15.36,
                    15.47, 15.57, 15.67, 15.76, 15.84, 15.92, 15.99, 16.06, 16.13, 16.19,
                    16.25, 16.30, 16.35, 16.40, 16.45, 16.49, 16.53, 16.57, 16.61, 16.64,
                    16.68, 16.71, 16.74, 16.77, 16.79, 16.82, 16.84, 16.87, 16.89, 16.91],
        "color"  : "#3498db",
    },
    "LSTM + Attention": {
        "epochs" : list(range(1, 51)),
        "loss"   : [3.45, 3.31, 3.22, 3.14, 3.08, 3.03, 2.99, 2.95, 2.91, 2.95,
                    2.91, 2.88, 2.85, 2.83, 2.81, 2.79, 2.77, 2.75, 2.74, 2.61,
                    2.53, 2.50, 2.47, 2.44, 2.42, 2.40, 2.38, 2.35, 2.33, 2.31,
                    2.28, 2.26, 2.24, 2.22, 2.20, 2.18, 2.16, 2.14, 2.12, 2.10,
                    2.09, 2.07, 2.05, 2.03, 2.01, 1.99, 1.97, 1.95, 1.92, 1.89],
        "acc"    : [10.0, 12.0, 14.0, 15.5, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
                    23.0, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0,
                    28.5, 29.0, 29.5, 30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0,
                    33.3, 33.6, 33.9, 34.2, 34.5, 34.8, 35.1, 35.4, 35.7, 36.0,
                    36.2, 36.4, 36.6, 36.8, 37.0, 37.1, 37.2, 37.3, 37.4, 37.5],
        "color"  : "#2ecc71",
    }
}


def style_ax(ax, title):
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, color="white", fontsize=14, pad=10)
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)


# ── Image 1 — Loss + Accuracy Curves ──────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig1.patch.set_facecolor("#0f0f0f")
fig1.suptitle(
    "LSTM Music Generation — Training Curves",
    fontsize=18, fontweight="bold", color="white", y=1.02
)

for name, data in results.items():
    ax1.plot(data["epochs"], data["loss"],
             color=data["color"], linewidth=2.5,
             marker="o", markersize=5, label=name)

for name, data in results.items():
    ax2.plot(data["epochs"], data["acc"],
             color=data["color"], linewidth=2.5,
             marker="o", markersize=5, label=name)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")

style_ax(ax1, "Loss Over Epochs")
style_ax(ax2, "Accuracy Over Epochs")

plt.tight_layout()
plt.savefig("chart_curves.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("✅ Saved: chart_curves.png")


# ── Image 2 — Bar Charts ───────────────────────────────
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
fig2.patch.set_facecolor("#0f0f0f")
fig2.suptitle(
    "LSTM Music Generation — Final Comparison",
    fontsize=18, fontweight="bold", color="white", y=1.02
)

names      = list(results.keys())
best_loss  = [min(d["loss"]) for d in results.values()]
best_acc   = [max(d["acc"])  for d in results.values()]
colors     = [d["color"]     for d in results.values()]

bars1 = ax3.bar(names, best_loss, color=colors,
                width=0.5, edgecolor="white", linewidth=0.5)
bars2 = ax4.bar(names, best_acc,  color=colors,
                width=0.5, edgecolor="white", linewidth=0.5)

for bar, val in zip(bars1, best_loss):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.05,
             f"{val:.2f}", ha="center",
             color="white", fontsize=11, fontweight="bold")

for bar, val in zip(bars2, best_acc):
    ax4.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             f"{val:.1f}%", ha="center",
             color="white", fontsize=11, fontweight="bold")

ax3.set_ylabel("Loss")
ax4.set_ylabel("Accuracy (%)")
ax3.set_xticklabels(names, fontsize=9)
ax4.set_xticklabels(names, fontsize=9)

style_ax(ax3, "Best Loss Per Model")
style_ax(ax4, "Best Accuracy Per Model")

plt.tight_layout()
plt.savefig("chart_bars.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f0f")
plt.show()
print("✅ Saved: chart_bars.png")