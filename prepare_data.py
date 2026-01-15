import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# STEP 1: Load the dataset
df = pd.read_csv("data/soil_data.csv")

# STEP 2: Define input features and target
features = [
    "Bulk_Density",
    "alt_BD",
    "Water_Content",
    "pH",
    "EC",
    "Soil_Respiration",
    "Bglucosidase",
    "Bglucosaminidase",
    "Alkaline Phosphatase",
    "Acid Phosphatase",
    "POX_C",
    "ACE"
]

target = "SOC_pct"

# STEP 3: Keep only selected columns
df = df[features + [target]]

# STEP 4: Remove rows where target is missing
df = df.dropna(subset=[target])

print("Rows after dropping missing target:", df.shape[0])

# STEP 5: Fill missing feature values with column mean
df[features] = df[features].fillna(df[features].mean())

# STEP 6: Split inputs and output
X = df[features]
y = df[target]

# STEP 7: Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# STEP 9: Save prepared datasets
pd.DataFrame(X_train, columns=features).to_csv("data/X_train.csv", index=False)
pd.DataFrame(X_test, columns=features).to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Phase 2 complete: prepared datasets saved.")
