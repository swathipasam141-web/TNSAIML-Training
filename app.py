import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Hyderabad House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load model and feature names
model = joblib.load("model.joblib")
feature_names = joblib.load("feature_names.joblib")

# Load Hyderabad dataset for dropdown options
df = pd.read_csv("dataset/Hyderbad_House_price.csv")

df["BHK"] = df["title"].str.extract(
    r"(\d+)\s*BHK"
).astype(float)

df["property_type"] = df["title"].str.replace(
    r"^\s*\d+\s*BHK\s*", "", regex=True
).str.strip()

# Application title
st.title("🏠 Hyderabad House Price Prediction")
st.write("Predict property prices in different areas of Hyderabad.")

# Sidebar
st.sidebar.header("Property Information")

# Property type
property_type = st.sidebar.selectbox(
    "Property Type",
    ["Apartment", "Independent House"]
)

# Hyderabad location
locations = sorted(df["location"].dropna().unique())

location = st.sidebar.selectbox(
    "Select Location",
    locations
)

# Property area
area = st.sidebar.number_input(
    "Area (Square Feet)",
    min_value=100,
    max_value=100000,
    value=1200
)

# Bedrooms
bhk = st.sidebar.selectbox(
    "Number of Bedrooms (BHK)",
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    index=1
)

# Building status
building_status = st.sidebar.selectbox(
    "Building Status",
    sorted(df["building_status"].dropna().unique())
)

# Create input DataFrame
input_data = {
    "location": location,
    "property_type": property_type,
    "building_status": building_status,
    "area_insqft": area,
    "BHK": float(bhk)
}

input_df = pd.DataFrame([input_data])

# Match model feature order
input_df = input_df[feature_names]

# Display property details
st.subheader("Property Details")

st.dataframe(
    input_df,
    use_container_width=True,
    hide_index=True
)

# Prediction
if st.button("Predict Price", type="primary"):

    try:
        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        st.success(
            f"Estimated House Price: ₹ {prediction:,.2f} Lakhs"
        )

        st.metric(
            "Estimated Price in Crores",
            f"₹ {prediction / 100:.2f} Crores"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption(
    "Note: Prices are estimates based on the available Hyderabad dataset."
)


# ==============================
# MODEL RESULTS & VISUALIZATION
# ==============================

st.divider()

st.header("📊 Model Results & Visualization")

# Load dataset
chart_df = pd.read_csv("dataset/Hyderbad_House_price.csv")

# Clean data
chart_df.columns = chart_df.columns.str.strip()

# Graph 1: Average Price by Location
st.subheader("1. Average House Price by Location")

location_prices = (
    chart_df.groupby("location")["price(L)"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(location_prices)

# Graph 2: Property Type Comparison
st.subheader("2. Property Type Comparison")

chart_df["property_type"] = chart_df["title"].str.replace(
    r"^\s*\d+\s*BHK\s*", "", regex=True
).str.strip()

type_prices = chart_df.groupby("property_type")["price(L)"].mean()

st.bar_chart(type_prices)

# Graph 3: Area vs Price
st.subheader("3. Area vs House Price")

st.scatter_chart(
    chart_df,
    x="area_insqft",
    y="price(L)",
    color="property_type"
)

# Graph 4: Model Performance
st.subheader("4. Model Performance")

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Extract BHK before selecting columns
chart_df["BHK"] = chart_df["title"].str.extract(
    r"(\d+)\s*BHK"
).astype(float).fillna(0)

# Select input features
X = chart_df[
    [
        "location",
        "property_type",
        "building_status",
        "area_insqft",
        "BHK"
    ]
].copy()

X["BHK"] = chart_df["BHK"]

y = chart_df["price(L)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

col1, col2 = st.columns(2)

col1.metric("Testing R² Score", f"{r2:.2%}")
col2.metric("Mean Absolute Error", f"₹{mae:.2f} Lakhs")

# Graph 5: Actual vs Predicted
st.subheader("5. Actual vs Predicted Prices")

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

st.scatter_chart(
    comparison,
    x="Actual Price",
    y="Predicted Price"
)
