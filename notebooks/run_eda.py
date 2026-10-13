import sys
sys.path.insert(0, "src")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import CLEAN_CSV, LABELS, RESULTS_DIR

sns.set_theme(style="whitegrid", palette="muted")
df = pd.read_csv(CLEAN_CSV)
print(f"Rows: {len(df)}")
print(df.head(3))

RESULTS_DIR.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

counts = df["label"].value_counts()
colors = {"NEG":"#ef4444", "NTL":"#94a3b8", "POS":"#22c55e"}
axes[0].bar(counts.index, counts.values, color=[colors[l] for l in counts.index])
axes[0].set_title("Label Distribution"); axes[0].set_xlabel("Label"); axes[0].set_ylabel("Count")
for i, (lbl, cnt) in enumerate(counts.items()):
    axes[0].text(i, cnt + 50, f"{cnt:,}\n({100*cnt/len(df):.1f}%)", ha="center", fontsize=9)

df["text_len"] = df["clean_text"].str.split().str.len()
axes[1].hist(df["text_len"], bins=40, color="#6c63ff", edgecolor="white")
axes[1].set_title("Text Length (words)"); axes[1].set_xlabel("Word count"); axes[1].set_ylabel("Frequency")

axes[2].hist(df["cmi"], bins=30, color="#f59e0b", edgecolor="white")
axes[2].set_title("Code-Mixing Index (CMI)"); axes[2].set_xlabel("CMI"); axes[2].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("results/eda_plots.png", dpi=150)
print("Saved results/eda_plots.png")

# CMI stats per label
print("\nCMI stats per label:")
print(df.groupby("label")["cmi"].describe().round(2))

# Low-CMI rows
low_cmi = df[df["cmi"] < 5]
print(f"\nLow-CMI rows (<5): {len(low_cmi)} (likely non-Telugu/Hindi/Indonesian)")

# Mention leakage
has_mention = df["clean_text"].str.contains("@user").sum()
print(f"Rows with @user: {has_mention} ({100*has_mention/len(df):.1f}%)")
