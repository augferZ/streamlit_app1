import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv("VN_housing/VN_housing_dataset.csv")  # sửa đường dẫn phù hợp

df = df.drop(columns=['Địa chỉ', 'Ngày'], errors='ignore')


# rename cho đúng với pipeline ở trên
df = df.rename(columns={
    "Địa chỉ": "address",
    "Quận": "district",
    "Huyện": "province",
    "Loại hình nhà ở": "type",
    "Giấy tờ pháp lý": "legal",
    "Số tầng": "floor",
    "Số phòng ngủ": "bedrooms",
    "Diện tích": "area",
    "Giá/m2": "price_per_m2",
})


import re

def extract_price(text):
    if not text or pd.isna(text):
        return None
    
    text = str(text)
    # Tìm số
    match = re.search(r'([\d,.]+)', text)
    if not match:
        return None
    
    # Xử lý số
    num = float(match.group(1).replace('.', '').replace(',', '.'))
    
    # Nhân với triệu nếu cần
    return int(num * 1000000 if 'triệu' in text else num)

# Sử dụng
df["price_per_m2"] = df["price_per_m2"].apply(extract_price)
df["area"] = df["area"].apply(extract_price)


# nếu bạn muốn dùng "price" thì tạo cột price mới
df["price"] = df["price_per_m2"] * df["area"]

df["bedrooms"] = df["bedrooms"].replace("nhiều hơn 10 phòng",11)
df["bedrooms"] = df["bedrooms"].apply(extract_price)
df["floor"] = df["floor"].replace("Nhiều hơn 10",11)
df["floor"] = df["floor"].replace(np.nan,0).astype(int)
df["legal"] = df["legal"].replace(np.nan,"Chưa có số")

df["district"] = df["district"].astype(str).str.lower().str.strip()
df["province"] = df["province"].astype(str).str.lower().str.strip()

def clean_price_outliers(group, col="price", factor=1.5):
    Q1 = group[col].quantile(0.25)
    Q3 = group[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return group[(group[col] >= lower) & (group[col] <= upper)]

df = df.groupby(["province", "district"], group_keys=False).apply(clean_price_outliers)

df["district_avg_price"] = df.groupby(["province", "district"])["price"].transform("median")
df["district_price_ratio"] = df["price"] / df["district_avg_price"]



# drop các cột không dùng
df = df.drop(columns=[
    "Unnamed: 0",
    "Dài",
    "Rộng",
    "price_per_m2"   # vì đã tạo price ở trên
], errors="ignore")

mlib.cr_log("read",df)
mlib.pr_info(df)

# xem lại các cột sau xử lý
print(df.columns)
df.head()

#=======================================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ====== CHỌN FEATURE ======
features = [
    "province",
    "district",
    "type",
    "legal",
    "floor",
    "bedrooms",
    "area",
    "district_avg_price",
    "district_price_ratio"
]

X = df[features]
y = df["price"]

# ====== SPLIT ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====== ENCODER ======
from sklearn.impute import SimpleImputer

cat_cols = ["province", "district", "type", "legal"]
num_cols = [
    "floor", "bedrooms", "area",
    "district_avg_price", "district_price_ratio"
]

preprocess = ColumnTransformer([
    ("cat", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols),

    ("num", Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]), num_cols)
])


model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)



# ==== Sau khi train xong, tạo geo_stats.pkl ====
geo_stats = df.groupby(["province", "district"])["price"].agg(["median"]).reset_index()
geo_stats = geo_stats.rename(columns={"median": "district_avg_price"})
geo_stats["district_price_ratio"] = 1.0  # ratio mặc định =1

# Lưu file pickle
joblib.dump(geo_stats, "geo_stats.pkl")
print("Saved geo_stats.pkl cùng lúc với train model")

# ==== Xuất luôn model để deploy ====
joblib.dump(model, "house_price_model.pkl")
print("Saved trained model house_price_model.pkl")

#==============================================================
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("MAE:", mae)
print("R²:", r2)

import joblib

joblib.dump(model, "house_price_model.pkl")
print("Saved model to house_price_model.pkl")

