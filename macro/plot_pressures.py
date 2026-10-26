from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Folder containing the CSV files
folder = Path(".")

files = sorted(folder.glob("outlet_flux*.csv"))

series = {
    "coupling": {
        "label": "with coupling",
        "linestyle": "-",
        "marker": "o",
        "points": [],
    },
    "analytical": {
        "label": "with cubic law",
        "linestyle": "--",
        "marker": "s",
        "points": [],
    },
}

for file in files:
    case = next((name for name in series if name in file.stem.lower()), None)
    if case is None:
        continue

    df = pd.read_csv(file)
    if df.empty:
        continue

    final_value = df.iloc[-1]
    series[case]["points"].append(
        (
            float(final_value["bc_pressure_difference"]),
            float(final_value["outlet_fluid_flux"]),
        )
    )

fig, ax = plt.subplots(figsize=(7.5, 5))

for values in series.values():
    points = sorted(values["points"])
    if not points:
        continue

    pressure_differences, outlet_fluxes = zip(*points)
    ax.plot(
        pressure_differences,
        outlet_fluxes,
        linestyle=values["linestyle"],
        marker=values["marker"],
        linewidth=2,
        label=values["label"],
    )

ax.set_xlabel("BC pressure difference (Pa)")
ax.set_ylabel("Outlet fluid flux")

ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

out = "p_comparison.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
