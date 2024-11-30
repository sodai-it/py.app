import streamlit as st
import numpy as np
import pandas as pd

st.page_link("app.py", label="Home", icon="🏠")

# タイトル
st.title("5教科の成績アプリ")

# 10人生徒の５教科の成績
df = pd.DataFrame(np.random.randint(50, 100, (10, 5)), index=("伊藤", "佐々木", "山田", "田中", "斎藤", "山口", "後藤", "飯田", "朝倉", "大野"), columns=("国語", "数学", "理科", "社会", "英語"))
df["合計"] = df.sum(axis=1)
st.write(df)

# 棒グラフ
st.title("成績表")
df["総合得点"] = df["国語"] + df["数学"] + df["理科"] + df["社会"] + df["英語"]
st.bar_chart(df["総合得点"])

# 折れ線グラフ
st.title("成績表")
df["総合得点"] = df["国語"] + df["数学"] + df["理科"] + df["社会"] + df["英語"]
st.line_chart(df["総合得点"])

# 散布図
st.title("成績表")
df["総合得点"] = df["国語"] + df["数学"] + df["理科"] + df["社会"] + df["英語"]
st.scatter_chart(df["総合得点"])

# Mapに散布図表示
st.title("大阪府付近に散布図")
map_df = pd.DataFrame(
    np.random.rand(50,2)/[50, 50] + [34.69, 135.50],
    columns=["lat", "lon"]
)

st.map(map_df)


