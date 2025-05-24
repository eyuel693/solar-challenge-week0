import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


df = pd.read_csv("../data/sierraleone-bumbuna.csv")  
print("Initial shape:", df.shape)


print("\nSummary statistics:")
print(df.describe())


print("\nMissing values:")
print(df.isna().sum())


key_cols = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']
df.dropna(subset=key_cols, inplace=True)


for col in key_cols:
    df = df[np.abs(zscore(df[col])) < 3]


df.fillna(df.median(numeric_only=True), inplace=True)


df['Timestamp'] = pd.to_datetime(df['Timestamp'])


plt.figure(figsize=(12, 5))
plt.plot(df['Timestamp'], df['GHI'])
plt.title('GHI over Time - Sierra Leone')
plt.xlabel('Date')
plt.ylabel('GHI (W/m²)')
plt.grid()
plt.tight_layout()
plt.savefig("sierra_ghi_over_time.png")


df.groupby('Cleaning')[['ModA', 'ModB']].mean().plot(kind='bar')
plt.title('ModA & ModB - Cleaning Impact (Sierra Leone)')
plt.savefig("sierra_cleaning_impact.png")


sns.heatmap(df[['GHI', 'DNI', 'DHI', 'TModA', 'TModB']].corr(), annot=True)
plt.title('Correlation Heatmap - Sierra Leone')
plt.savefig("sierra_correlation_heatmap.png")


sns.scatterplot(data=df, x='WS', y='GHI')
plt.title('Wind Speed vs GHI - Sierra Leone')
plt.savefig("sierra_ws_vs_ghi.png")


df.to_csv("../data/sierra_leone_clean.csv", index=False)

print("✅ EDA and cleaning complete for Sierra Leone. Cleaned data saved.")

