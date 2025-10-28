# quick_check_dynamic.py
import pandas as pd, numpy as np, matplotlib.pyplot as plt
df = pd.read_csv("outputs/f1_dynamic/dcsi_race.csv")
df["pred_rank"] = df.groupby("race_id")["pred_perf"].rank(ascending=False, method="average")
plt.scatter(df["pred_rank"], df["finish_position"], s=6, alpha=0.3); plt.gca().invert_yaxis()
plt.xlabel("Pred rank"); plt.ylabel("Finish pos"); plt.title("Dynamic: Pred rank vs finish"); plt.show()
