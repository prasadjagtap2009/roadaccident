import pandas as pd
import os

# ---------- CONFIGURATION ----------
base_path = r"D:\DM micro"
file_path = os.path.join(base_path, "Acc.csv")

print(f"\nLoading FULL dataset from: {file_path}")
print("Please wait...")

# ---------- 1. LOAD DATA ----------
df = pd.read_csv(file_path, low_memory=False)

print(f"Original Dataset Rows: {len(df)}")
print(f"Original Columns: {len(df.columns)}")

# ---------- 2. REMOVE DUPLICATES ----------
if 'Accident_Index' in df.columns:
    df.drop_duplicates(subset=['Accident_Index'], inplace=True)

print(f"After removing duplicates: {len(df)} rows")

# ---------- 3. SELECT USEFUL COLUMNS ----------
useful_cols = [
    'Day_of_Week',
    'Light_Conditions',
    'Weather_Conditions',
    'Road_Surface_Conditions',
    'Speed_limit',
    'Urban_or_Rural_Area',
    'Number_of_Vehicles',
    'Number_of_Casualties',
    'Accident_Severity'
]

# Keep only columns that exist in dataset
existing_cols = [c for c in useful_cols if c in df.columns]

df_final = df[existing_cols].copy()

print(f"Columns used for analysis: {existing_cols}")

# ---------- 4. HANDLE MISSING VALUES ----------

print("Cleaning missing values...")

# Convert categorical columns to string
categorical_cols = [
    'Day_of_Week',
    'Light_Conditions',
    'Weather_Conditions',
    'Road_Surface_Conditions',
    'Urban_or_Rural_Area'
]

for col in categorical_cols:
    if col in df_final.columns:
        df_final[col] = df_final[col].astype(str)
        df_final[col] = df_final[col].replace('nan', 'Unknown')

# Fill numeric missing values with 0
numeric_cols = [
    'Speed_limit',
    'Number_of_Vehicles',
    'Number_of_Casualties',
    'Accident_Severity'
]

for col in numeric_cols:
    if col in df_final.columns:
        df_final[col] = df_final[col].fillna(0)

# ---------- 5. FINAL DATASET INFO ----------

print("\nFinal dataset info:")
print(df_final.info())

# ---------- 6. SAVE CLEAN DATASET ----------

output_file = os.path.join(base_path, "India_Accidents_Cleaned_WEKA.csv")

df_final.to_csv(output_file, index=False)

print("\nSUCCESS ====================================")
print(f"Final Dataset Rows: {len(df_final)}")
print(f"Dataset saved to: {output_file}")
print("You can now open this dataset in WEKA.")
print("=============================================")