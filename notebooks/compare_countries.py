import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import f_oneway


benin = pd.read_csv("../data/benin_clean.csv")
sl = pd.read_csv("../data/sierra_leone_clean.csv")
togo = pd.read_csv("../data/togo_clean.csv")


benin["Country"] = "Benin"
sl["Country"] = "Sierra Leone"
togo["Country"] = "Togo"


df = pd.concat([benin, sl, togo], axis=0)


for col in ["GHI", "DNI", "DHI"]:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Country", y=col, data=df)
    plt.title(f"{col} Comparison Across Countries")
    plt.savefig(f"{col.lower()}_boxplot.png")


summary = df.groupby("Country")[["GHI", "DNI", "DHI"]].agg(["mean", "median", "std"])
print("\nSummary Table:")
print(summary)


ghi_p = f_oneway(benin["GHI"], sl["GHI"], togo["GHI"]).pvalue
print(f"\nANOVA test for GHI differences: p-value = {ghi_p:.4f}")
if ghi_p < 0.05:
    print("→ Statistically significant difference in GHI between countries.")
else:
    print("→ No statistically significant difference in GHI between countries.")


avg_ghi = df.groupby("Country")["GHI"].mean().sort_values(ascending=False)

plt.figure(figsize=(6, 4))
avg_ghi.plot(kind="bar", color=["#f4a261", "#2a9d8f", "#e76f51"])
plt.title("Average GHI by Country")
plt.ylabel("GHI (W/m²)")
plt.tight_layout()
plt.savefig("avg_ghi_bar.png")

print("\n Cross-country comparison complete. Plots saved.")

