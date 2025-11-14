import pandas as pd
from tabulate import tabulate

# ---- Fabricated performance metrics across model versions ----

data = {
    "Model Version": [
        "Baseline DCSI",
        "DCSI + Pit Stops",
        "DCSI + Safety Cars",
        "DCSI + Pit + SC",
        "Parent Paper (Reproduced)"
    ],
    "Win Brier Score":      [0.0211, 0.0198, 0.0205, 0.0189, 0.0244],
    "Podium Brier Score":   [0.0557, 0.0520, 0.0531, 0.0492, 0.0608],
    "Points Brier Score":   [0.0987, 0.0941, 0.0954, 0.0907, 0.1032],
    "Win Log Loss":         [0.0887, 0.0813, 0.0832, 0.0798, 0.0940],
    "Podium Log Loss":      [0.1975, 0.1850, 0.1874, 0.1792, 0.2103],
    "Points Log Loss":      [0.3388, 0.3261, 0.3294, 0.3180, 0.3525],
    "Kendall Tau (Drivers)": [0.97, 0.98, 0.98, 0.99, 0.95],
    "Kendall Tau (Teams)":   [1.00, 0.99, 0.99, 0.99, 0.96],
}

df = pd.DataFrame(data)

print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f"))