import os
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

st.header("Giysi Bedeni Tahmin Ekranı")

rad = st.sidebar.radio("Menu", ["Home", "RandomForestRegressor", "LinearRegression", "DecisionTreeRegressor", "GradientBoostingRegressor"])

if rad == "Home":
    st.subheader("Ana Sayfa")
    st.write("Lütfen sol taraftaki menüden bir model seçiniz.")
else:
    st.subheader(f"{rad} ile Beden Tahmini")

    # Göreceli dosya yolunu kullanarak CSV dosyasını okuma
    current_dir = os.path.dirname(__file__)  # Bu Python dosyasının bulunduğu klasörü alır
    file_path = os.path.join(current_dir, "Clothing.csv")  # CSV dosyasının tam yolunu oluşturur

    clothes = pd.read_csv(file_path)

    clothes = clothes.dropna()
    
    le = LabelEncoder()
    clothes['size'] = le.fit_transform(clothes['size'])

    X = clothes.drop(['size'], axis=1)
    y = clothes['size']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42)
    }
    
    model = models[rad]
    model.fit(X_train, y_train)

    weight = st.number_input("Ağırlık (kg):", min_value=22.0, max_value=136.0, step=1.0)
    age = st.number_input("Yaş:", min_value=0.0, max_value=117.0, step=1.0)
    height = st.number_input("Boy (cm):", min_value=137.16, max_value=193.04, step=1.0)

    if st.button("Beden Tahmini Yap"):
        input_data = [[weight, age, height]]
        prediction = model.predict(input_data)
        predicted_size_index = round(prediction[0])
        predicted_size = le.inverse_transform([predicted_size_index])[0]
        st.write(f"Tahmini Beden: {predicted_size}")
