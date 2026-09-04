from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Folder containing the CSV files
folder = Path(".")

# Find all files matching pressure_curve*.csv
files = sorted(folder.glob("pressure_curve*.csv"))

fig, ax = plt.subplots(figsize=(7.5, 5))

for file in files:
    df = pd.read_csv(file)

    ax.plot(
        df["x_m"],
        df["pressure_pa"],
        marker="o",
        label=file.stem,
    )

ax.set_xlabel("x_m")
ax.set_ylabel("Pressure (Pa)")
ax.set_title("Pressure vs x_m")

ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

out = "p_comparison.png"
fig.savefig(out, dpi=200, bbox_inches="tight")

plt.show()

print(f"Saved plot: {out}")
print(f"Plotted {len(files)} files:")
for file in files:
    print(f"  - {file.name}")