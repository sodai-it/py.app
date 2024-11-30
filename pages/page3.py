import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.page_link("app.py", label="Home", icon="🏠")

# タイトル
st.write("ポケモンの種類予想アプリ")

# サイドバーでデータをアップロード
st.sidebar.header("データをアップロード")
upload_file = st.sidebar.file_uploader("ポケモンのCSVファイルをアップロードして下さい", type=["csv"])

if upload_file:
    # データを読み込む
    df = pd.read_csv(upload_file)
    # データの表示
    st.subheader("データプレビュー")
    st.dataframe(df.head())
    
    # ユーザーが選べるタイプのリスト(Type1を使用)
    available_types = df['Type 1'].unique()  # Type 1のユニークな値を取得
    select_types = st.multiselect(
        "タイプを３つ選んでください",
        available_types,
        max_selections=3  # 最大3つ選択
    )
    
    # もし選択されたタイプが1つ以上あれば、フィルタリングしてランダムにポケモンを選ぶ
    if select_types:
        # 複数選択されたタイプに基づいてフィルタリング
        filtered_pokemon = df[df["Type 1"].isin(select_types)]
        
        if len(filtered_pokemon) > 0:
            random_pokemon = filtered_pokemon.sample(n=3)  # ランダムに３匹選択
            st.write(f"選ばれたポケモン (タイプ: {', '.join(select_types)}) : ")
            for _, pokemon in random_pokemon.iterrows():
                st.write(pokemon.to_frame().T)  # ポケモンのデータ行全体を表示
        else:
            st.write("選択されたタイプではポケモンはいません")
    else:
        st.write("ポケモンを選んでください")
    
    # ターゲット変数（y）は 'Type 1'
    if 'Type 1' in df.columns:
        y = df['Type 1'].astype('category').cat.codes  # Type 1を数値化

        # 特徴量（X）は 'Type 1' と 'Name' を除く数値カラム
        X = df.drop(columns=['Type 1', 'Name'])  # 'Type 1' と 'Name' を除外
        X = X.select_dtypes(include=['float64', 'int64'])  # 数値カラムのみ選択
        
        # 学習データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # モデルの学習
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        # テストデータの予測
        y_pred = model.predict(X_test)
        
        # 精度を計算
        accuracy = accuracy_score(y_test, y_pred) * 100  # パーセンテージ形式に変換
        st.write(f"モデルの精度は {accuracy:.2f}% です")
else:
    st.write("ポケモンのCSVファイルをアップロードしてください。")
