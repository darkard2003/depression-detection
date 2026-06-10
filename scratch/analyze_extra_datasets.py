import os
import pandas as pd

SHREYA_PATH = "datasets/extra/shreyar_depression_detection.csv"
OURAFLA_PATH = "datasets/extra/ourafla_mental_health.csv"

def analyze_shreya():
    print("=" * 60)
    print("ANALYZING: Shreya Depression Detection Dataset")
    print("=" * 60)
    if not os.path.exists(SHREYA_PATH):
        print(f"Error: {SHREYA_PATH} not found!")
        return
    df = pd.read_csv(SHREYA_PATH)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nLabel Distribution ('is_depression'):")
    print(df['is_depression'].value_counts(dropna=False))
    print("\nSplit Distribution ('split'):")
    print(df['split'].value_counts(dropna=False))

def analyze_ourafla():
    print("\n" + "=" * 60)
    print("ANALYZING: Ourafla Mental Health Dataset")
    print("=" * 60)
    if not os.path.exists(OURAFLA_PATH):
        print(f"Error: {OURAFLA_PATH} not found!")
        return
    df = pd.read_csv(OURAFLA_PATH)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nLabel Distribution ('status'):")
    print(df['status'].value_counts(dropna=False))

if __name__ == "__main__":
    analyze_shreya()
    analyze_ourafla()
