import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

#─── 1. Load Data ────────────────────────────────────────────────────────────
files = glob.glob("bsds_train_autoCanny_*.csv")
dfs = []
for f in files:
    sigma = float(os.path.basename(f).split("_")[-1].replace(".csv", ""))
    df = pd.read_csv(f)
    df["sigma"] = sigma
    dfs.append(df)
full_df = pd.concat(dfs)

#─── 2. Aggregate Statistics ─────────────────────────────────────────────────
agg_df = full_df.groupby("sigma").agg({
    "P": ["mean", "std"],
    "R": ["mean", "std"], 
    "F1": ["mean", "std"],
    "ms": ["mean", "std"]
}).reset_index()

#─── 3. Generate Plots ───────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

#─── Plot 1: Precision/Recall/F1 vs Sigma ────────────────────────────────────
metrics = ["P", "R", "F1"]
for metric in metrics:
    axs[0,0].errorbar(agg_df["sigma"], 
                     agg_df[metric]["mean"],
                     yerr=agg_df[metric]["std"],
                     label=metric, marker="o", capsize=5)
axs[0,0].set(xlabel="Sigma", ylabel="Score", 
            title="Edge Detection Performance vs Sigma")
axs[0,0].legend()
axs[0,0].grid(True, alpha=0.3)

#─── Plot 2: Processing Time vs Sigma ────────────────────────────────────────
axs[0,1].plot(agg_df["sigma"], agg_df["ms"]["mean"], 
             marker="o", color="purple")
axs[0,1].set(xlabel="Sigma", ylabel="Time (ms)", 
            title="Average Processing Time")
axs[0,1].grid(True, alpha=0.3)

#─── Plot 3: F1 Distribution by Sigma ────────────────────────────────────────
sns.boxplot(data=full_df, x="sigma", y="F1", ax=axs[1,0], 
           palette="viridis")
axs[1,0].set(xlabel="Sigma", ylabel="F1 Score", 
            title="F1 Score Distribution Across Images")
axs[1,0].tick_params(axis="x", rotation=45)

#─── Plot 4: Precision-Recall Tradeoff ───────────────────────────────────────
sns.scatterplot(data=agg_df, x=("R", "mean"), y=("P", "mean"),
               hue="sigma", palette="viridis", s=200, 
               ax=axs[1,1])
axs[1,1].set(xlabel="Recall", ylabel="Precision", 
            title="Precision-Recall Tradeoff")
axs[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sigma_performance_analysis.png", dpi=300, bbox_inches="tight")
plt.show()
