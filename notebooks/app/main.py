import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    benin = pd.read_csv("../data/benin_clean.csv")
    sierra = pd.read_csv("../data/sierra_leone_clean.csv")
    togo = pd.read_csv("../data/togo_clean.csv")
    
    benin["Country"] = "Benin"
    sierra["Country"] = "Sierra Leone"
    togo["Country"] = "Togo"

    return pd.concat([benin, sierra, togo], ignore_index=True)

df = load_data()


st.title("☀️ Solar Data Dashboard")
st.markdown("Interactive comparison of solar metrics across Benin, Sierra Leone, and Togo.")


metric = st.selectbox("Select Metric", ["GHI", "DNI", "DHI"])
plot_type = st.radio("Plot Type", ["Boxplot", "Bar Chart (Mean)"])

if plot_type == "Boxplot":
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Country", y=metric, ax=ax)
    st.pyplot(fig)

else:
    avg = df.groupby("Country")[metric].mean().sort_values(ascending=False)
    st.bar_chart(avg)


if st.checkbox("Show Raw Data"):
    st.write(df[["Timestamp", "Country", metric]])
