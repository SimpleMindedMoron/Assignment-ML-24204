import numpy as np
import pandas as pd


file_path = "Lab Session Data.xlsx" 
df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

df.replace("?", np.nan, inplace=True)


print("--- DATA TYPES AND ENCODING SCHEMES ---")


numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()


ordinal_cols = [] 
nominal_cols = [col for col in categorical_cols if col not in ordinal_cols]

for col in df.columns:
    if col in numeric_cols:
        dtype = "Numeric (Ratio/Interval)"
        encoding = "No Encoding (Scaling if needed)"
    elif col in ordinal_cols:
        dtype = "Categorical (Ordinal)"
        encoding = "Label Encoding"
    else:
        dtype = "Categorical (Nominal)"
        encoding = "One-Hot Encoding"

    print(f"Attribute: {col:20s} | Type: {dtype:25s} | Encoding: {encoding}")



print("\n--- NUMERIC VARIABLE RANGES ---")
for col in numeric_cols:
    col_min = df[col].min()
    col_max = df[col].max()
    print(f"Attribute: {col:15s} | Range: [{col_min} to {col_max}]")


print("\n--- MISSING VALUES PER ATTRIBUTE ---")
missing_info = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame(
    {"Missing Count": missing_info, "Percentage (%)": missing_percent.round(2)}
)
print(missing_df[missing_df["Missing Count"] > 0])




print("\n--- MEAN, VARIANCE, AND STANDARD DEVIATION ---")
stats_df = pd.DataFrame(
    {
        "Mean": df[numeric_cols].mean(),
        "Variance": df[numeric_cols].var(),
        "Std Dev": df[numeric_cols].std(),
    }
)
print(stats_df.round(4))

