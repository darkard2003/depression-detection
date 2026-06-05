#!/usr/bin/env python3
import os
import re

ROOT_DIR = "/Users/dark/code/project/depression"

def update_root_script(filepath):
    print(f"Updating root script: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replacements for datasets
    content = content.replace('"bin_reddit1.csv"', '"datasets/bin_reddit1.csv"')
    content = content.replace("'bin_reddit1.csv'", "'datasets/bin_reddit1.csv'")
    content = content.replace('"thepixel42_depression-detection.csv"', '"datasets/thepixel42_depression-detection.csv"')
    content = content.replace("'thepixel42_depression-detection.csv'", "'datasets/thepixel42_depression-detection.csv'")

    # Replacements for processed directories
    content = re.sub(r'(["\'])(processed_chi2)(["\'])', r'\1data_processed/processed_chi2\3', content)
    content = re.sub(r'(["\'])(processed_bert)(["\'])', r'\1data_processed/processed_bert\3', content)
    content = re.sub(r'(["\'])(processed_dirty)(["\'])', r'\1data_processed/processed_dirty\3', content)
    # Be careful not to replace things like "data_processed" with "data_data_processed"
    content = re.sub(r'(?<!data_)(["\'])(processed)(["\'])(?!_chi2|_bert|_dirty)', r'\1data_processed/processed\3', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def update_notebook(filepath):
    print(f"Updating notebook: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replacements for datasets
    content = content.replace('bin_reddit1.csv', '../datasets/bin_reddit1.csv')
    content = content.replace('thepixel42_depression-detection.csv', '../datasets/thepixel42_depression-detection.csv')

    # Replacements for processed directories (with ../ prefix)
    content = re.sub(r'(["\'])(processed_chi2)(["\'])', r'\1../data_processed/processed_chi2\3', content)
    content = re.sub(r'(["\'])(processed_bert)(["\'])', r'\1../data_processed/processed_bert\3', content)
    content = re.sub(r'(["\'])(processed_dirty)(["\'])', r'\1../data_processed/processed_dirty\3', content)
    content = re.sub(r'(?<!data_)(["\'])(processed)(["\'])(?!_chi2|_bert|_dirty)', r'\1../data_processed/processed\3', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def update_makefile(filepath):
    print(f"Updating Makefile: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.replace('--data_dir processed_chi2', '--data_dir data_processed/processed_chi2')
    content = content.replace('--data_dir processed_dirty', '--data_dir data_processed/processed_dirty')
    content = content.replace('--data_dir processed_bert', '--data_dir data_processed/processed_bert')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    # 1. Update root python scripts
    for file in os.listdir(ROOT_DIR):
        if file.endswith('.py') and file != 'check_gpu.py':
            update_root_script(os.path.join(ROOT_DIR, file))

    # 2. Update scratch python scripts
    scratch_dir = os.path.join(ROOT_DIR, 'scratch')
    if os.path.exists(scratch_dir):
        for file in os.listdir(scratch_dir):
            if file.endswith('.py') and file != 'update_paths.py':
                update_root_script(os.path.join(scratch_dir, file))

    # 3. Update notebooks
    notebooks_dir = os.path.join(ROOT_DIR, 'notebooks')
    if os.path.exists(notebooks_dir):
        for file in os.listdir(notebooks_dir):
            if file.endswith('.ipynb'):
                update_notebook(os.path.join(notebooks_dir, file))

    # 4. Update Makefile
    makefile_path = os.path.join(ROOT_DIR, 'Makefile')
    if os.path.exists(makefile_path):
        update_makefile(makefile_path)

    print("Path updates complete!")

if __name__ == '__main__':
    main()
