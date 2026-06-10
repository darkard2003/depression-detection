#!/usr/bin/env python3
import os
import sys
import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = "datasets/extra"

def download_shreyar():
    print("Downloading ShreyaR/DepressionDetection...")
    try:
        dataset = load_dataset("ShreyaR/DepressionDetection")
        dfs = []
        for split in dataset.keys():
            df_split = pd.DataFrame(dataset[split])
            df_split['split'] = split
            dfs.append(df_split)
        df = pd.concat(dfs, ignore_index=True)
        
        output_path = os.path.join(OUTPUT_DIR, "shreyar_depression_detection.csv")
        df.to_csv(output_path, index=False)
        print(f"✓ Saved ShreyaR/DepressionDetection to '{output_path}'. Shape: {df.shape}")
    except Exception as e:
        print(f"Error downloading ShreyaR/DepressionDetection: {e}")

from huggingface_hub import hf_hub_download
import shutil

def download_ourafla():
    print("Downloading ourafla/Mental-Health_Text-Classification_Dataset directly...")
    try:
        # Download raw unbalanced csv file directly from HF Hub
        file_path = hf_hub_download(
            repo_id="ourafla/Mental-Health_Text-Classification_Dataset",
            filename="mental_heath_unbanlanced.csv",
            repo_type="dataset"
        )
        output_path = os.path.join(OUTPUT_DIR, "ourafla_mental_health.csv")
        shutil.copy(file_path, output_path)
        print(f"✓ Saved ourafla/Mental-Health_Text-Classification_Dataset directly to '{output_path}'")
    except Exception as e:
        print(f"Error downloading ourafla dataset: {e}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    download_shreyar()
    print("-" * 50)
    download_ourafla()

if __name__ == "__main__":
    main()
