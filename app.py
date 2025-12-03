import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt

# ===== Load model và geo_stats =====
model = joblib.load("house_price_model.pkl")
geo_stats = joblib.load("geo_stats.pkl")  # vẫn lưu trung bình giá theo district

st.set_page_config(page_title="Dự đoán giá nhà Hà Nội", layout="wide")
st.title("🏠 Dự đoán giá nhà ở Hà Nội")
st.write("Nhập thông tin để dự đoán giá nhà tại Hà Nội")

# ===== Quận (district) =====
district_list = geo_stats["district"].sort_values().unique().tolist()
district_list = [text.title() for text in district_list]
district = st.selectbox("Quận", district_list).lower()

# ===== Phường (province) =====
# Lấy phường trong quận đã chọn
wards_in_district = geo_stats[geo_stats["district"]==district]["province"].sort_values().unique().tolist()
wards_in_district = [text.title() for text in wards_in_district]
province = st.selectbox("Phường", wards_in_district).lower()

# Hỗ trợ nhập phường mới
province_input = st.text_input("Hoặc nhập Phường khác nếu không có trong danh sách", "")
if province_input.strip() != "":
    province = province_input.lower().strip()

# ===== Các input khác =====
type_ = st.selectbox("Loại nhà", ["nhà riêng", "nhà mặt phố", "chung cư", "khác"])
legal = st.selectbox("Giấy tờ pháp lý", ["sổ đỏ", "sổ hồng", "hợp đồng", "khác"])
floor = st.number_input("Số tầng", min_value=0, max_value=50, value=1)
bedrooms = st.number_input("Số phòng ngủ", min_value=0, max_value=20, value=1)
area = st.number_input("Diện tích (m2)", min_value=5.0, max_value=3000.0, value=50.0)

# ===== Lấy district_avg_price từ geo_stats =====
row = geo_stats[geo_stats["district"] == district]

if not row.empty:
    # district_avg_price = float(row["district_avg_price"])
    district_avg_price = float(row["district_avg_price"].iloc[0])
    # district_price_ratio = float(row["district_price_ratio"])
    district_price_ratio = float(row["district_price_ratio"].iloc[0])
else:
    # fallback nếu district mới
    district_avg_price = geo_stats["district_avg_price"].median()
    district_price_ratio = 1.0

# ===== Build input DataFrame =====
input_data = pd.DataFrame([{
    "province": province,
    "district": district,
    "type": type_,
    "legal": legal,
    "floor": floor,
    "bedrooms": bedrooms,
    "area": area,
    "district_avg_price": district_avg_price,
    "district_price_ratio": district_price_ratio
}])

# ===== Predict =====
if st.button("Dự đoán giá"):
    price = model.predict(input_data)[0]
    mae = 18935472  # từ model train
    lower = price - mae
    upper = price + mae

    st.success(f"💰 Giá dự đoán: {price:,.0f} VNĐ")
    st.info(f"📉 Dải giá dự đoán ±MAE = ( {mae:,.0f} ): {lower:,.0f} - {upper:,.0f} VNĐ")

    # ===== Biểu đồ so sánh giá trung bình quận =====
    avg_price = geo_stats.loc[geo_stats['district']==district, 'district_avg_price'].values[0]
    fig, ax = plt.subplots()
    ax.bar(["Giá trung bình quận", "Giá dự đoán"], [avg_price, price], color=["skyblue", "orange"])
    ax.set_ylabel("VNĐ")
    ax.set_title(f"So sánh giá dự đoán với giá trung bình quận {district.title()}")
    st.pyplot(fig)

    # ===== Map visualization =====
    st.subheader("📍 Vị trí quận trên bản đồ (giả lập)")
    df_map = pd.DataFrame({
        'lat': [21.0278],  # trung tâm Hà Nội (ví dụ)
        'lon': [105.8342],
        'price': [price]
    })
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=21.0278,
            longitude=105.8342,
            zoom=11,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df_map,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=500,
                pickable=True
            )
        ]
    ))
