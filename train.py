import pandas as pd
import joblib
import re

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Step 1: Read Hyderabad dataset
data = Path("dataset/Hyderbad_House_price.csv")
df = pd.read_csv(data)

# Step 2: Data cleaning
df.columns = df.columns.str.strip()

df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df = df.drop_duplicates()
df = df.dropna()

# Step 3: Extract BHK and property type
df["title"] = df["title"].str.strip()

df["BHK"] = df["title"].str.extract(r"(\d+)\s*BHK").astype(float)
df["BHK"] = df["BHK"].fillna(0)

df["property_type"] = df["title"].str.replace(
    r"^\s*\d+\s*BHK\s*", "", regex=True
).str.strip()

# Keep residential buildings only
valid_types = [
    "Apartment",
    "Independent House",
    "Villa",
    "Independent Floor",
    "Studio Apartment"
]

df = df[df["property_type"].isin(valid_types)].copy()

# Step 4: Select features and target
X = df[
    [
        "location",
        "property_type",
        "building_status",
        "area_insqft",
        "BHK"
    ]
]

y = df["price(L)"]

# Step 5: Encode categorical columns
categorical_features = [
    "location",
    "property_type",
    "building_status"
]

numerical_features = [
    "area_insqft",
    "BHK"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features),
        ("num", "passthrough", numerical_features)
    ]
)

# Step 6: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 7: Train model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=150,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# Step 8: Evaluate model
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Training R2 Score:", round(r2_score(y_train, train_pred), 4))
print("Testing R2 Score:", round(r2_score(y_test, test_pred), 4))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, test_pred), 2), "Lakhs")

# Step 9: Save model
joblib.dump(model, "model.joblib")
joblib.dump(X.columns.tolist(), "feature_names.joblib")

print("Model trained and saved successfully!")
