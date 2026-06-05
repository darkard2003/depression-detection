#!/usr/bin/env python3
import os
from datasets import load_dataset

def main():
    print("=" * 60)
    print("Downloading thePixel42/depression-detection from Hugging Face")
    print("=" * 60)
    
    try:
        # Load dataset
        dataset = load_dataset("thePixel42/depression-detection")
        
        # Convert train split to Pandas
        df = dataset['train'].to_pandas()
        
        # Output file name
        output_csv = "datasets/thepixel42_depression-detection.csv"
        
        # Save to CSV
        df.to_csv(output_csv, index=False)
        print(f"✓ Successfully saved dataset ({len(df)} rows) to '{output_csv}'")
    except Exception as e:
        print(f"Error downloading or saving dataset: {e}")
        
    print("=" * 60)

if __name__ == '__main__':
    main()
