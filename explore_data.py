import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1️⃣ Load the dataset
data_path = "data/soil_data.csv"  # adjust if your file has a different name
df = pd.read_csv(data_path)

# 2️⃣ Quick look at the first 5 rows
print("First 5 rows:")
print(df.head())

# 3️⃣ Dataset info
print("\nDataset info:")
print(df.info())

# 4️⃣ Descriptive statistics
print("\nBasic statistics:")
print(df.describe())

# 5️⃣ Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# 6️⃣ Correlation heatmap (Numeric columns only)
plt.figure(figsize=(12, 10))  # Increased size for better visibility

# Select only numeric columns first
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Calculate correlation on numeric data only
sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm") # annot=False is better for 50+ columns
plt.title("Correlation between soil parameters")
plt.show()

# 7️⃣ Histograms of numeric columns
df.hist(bins=20, figsize=(12,8))
plt.show()

