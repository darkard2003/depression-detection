#!/usr/bin/env python3
import os
import pandas as pd

THEPIXEL42_PATH = "datasets/thepixel42_depression-detection.csv"
SHREYA_PATH = "datasets/extra/shreyar_depression_detection.csv"
OURAFLA_PATH = "datasets/extra/ourafla_mental_health.csv"
OUTPUT_PATH = "datasets/combined_dataset.csv"

def main():
    print("=" * 60)
    print("Merging datasets and mapping labels...")
    print("=" * 60)
    
    # 1. Load thePixel42
    print(f"Loading main dataset: {THEPIXEL42_PATH}...")
    df1 = pd.read_csv(THEPIXEL42_PATH)
    print(f"thePixel42 shape: {df1.shape}")
    print("thePixel42 label distribution:")
    print(df1['label'].value_counts(dropna=False))
    
    # Keep only text and label
    df1_clean = pd.DataFrame({
        'text': df1['text'].fillna("").astype(str),
        'label': df1['label'].astype(int),
        'source': 'thepixel42'
    })

    # 2. Load Shreya
    print(f"\nLoading Shreya dataset: {SHREYA_PATH}...")
    df2 = pd.read_csv(SHREYA_PATH)
    print(f"Shreya shape: {df2.shape}")
    print("Shreya is_depression distribution:")
    print(df2['is_depression'].value_counts(dropna=False))
    
    # Map clean_text to text and is_depression to label
    df2_clean = pd.DataFrame({
        'text': df2['clean_text'].fillna("").astype(str),
        'label': df2['is_depression'].astype(int),
        'source': 'shreya'
    })

    # 3. Load Ourafla
    print(f"\nLoading Ourafla dataset: {OURAFLA_PATH}...")
    df3 = pd.read_csv(OURAFLA_PATH)
    print(f"Ourafla shape: {df3.shape}")
    print("Ourafla status distribution before filtering:")
    print(df3['status'].value_counts(dropna=False))
    
    # Exclude Anxiety class
    df3_filtered = df3[df3['status'] != 'Anxiety'].copy()
    print(f"Ourafla shape after excluding Anxiety: {df3_filtered.shape}")
    
    # Map Normal -> 0, Depression / Suicidal / Bipolar / ADHD / PTSD / etc. -> 1
    # Check what classes are left
    print("Remaining Ourafla classes:")
    print(df3_filtered['status'].value_counts(dropna=False))
    
    # Define label mapping: status == 'Normal' -> 0, all other classes -> 1
    labels_mapped = (df3_filtered['status'] != 'Normal').astype(int).values
    
    df3_clean = pd.DataFrame({
        'text': df3_filtered['text'].fillna("").astype(str),
        'label': labels_mapped,
        'source': 'ourafla'
    })

    # 4. Concatenate datasets
    print("\nConcatenating datasets...")
    df_combined = pd.concat([df1_clean, df2_clean, df3_clean], ignore_index=True)
    print(f"Combined dataset shape: {df_combined.shape}")
    print("Combined label distribution:")
    print(df_combined['label'].value_counts(dropna=False))
    print("Combined source distribution:")
    print(df_combined['source'].value_counts(dropna=False))
    
    # Save combined dataset
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved combined dataset to {OUTPUT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
