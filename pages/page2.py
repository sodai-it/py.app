import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.page_link("app.py", label="Home", icon="🏠")

st.title("●花予測アプリ")

iris = load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# 学習データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#　モデルの学習
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

#　テストデータの予測
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
st.write(f"精度は{accuracy:.2f}%です")

# ユーザー入力フォーム
st.header("好きな値を入力してください")
sepal_length = st.number_input("sepal length (cm)", min_value=0, value=3)
sepal_width = st.number_input("sepal width (cm)", min_value=0, value=3)
petal_length = st.number_input("petal length (cm)", min_value=0, value=3)
petal_width = st.number_input("petal width (cm)", min_value=0, value=3)

input_data = pd.DataFrame({
    "sepal length (cm)":[sepal_length],
    "sepal width (cm)":[sepal_width],
    "petal length (cm)":[petal_length],
    "petal width (cm)":[petal_width]
})

if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    species = iris.target_names[prediction][0]
    st.write(f"予測した品種は{species}です")
    
# データの可視化
st.header("データ可視化")
fig, ax = plt.subplots()
scatter = ax.scatter(x["petal length (cm)"], x["petal width (cm)"], c=y, label=iris.target_names)
ax.scatter(petal_length, petal_width, c="red")
ax.set_xlabel("petal length (cm)")
ax.set_ylabel("petal width (cm)")

handles, labels = scatter.legend_elements(prop="colors")
ax.legend(handles, iris.target_names, title="Species")
st.pyplot(fig)