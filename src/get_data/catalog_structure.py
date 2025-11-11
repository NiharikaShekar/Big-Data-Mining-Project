import os
import pandas as pd

ROOT = "data/raw/FakeNewsNet"

def summarize_folder(path):
    print(f"\n Folder: {path}")
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file.endswith(".csv"):

            #skip them empty lines
            if os.path.getsize(file_path) == 0:
                print (f" {file}: empty file, skipped")
                continue 
            try:
                df = pd.read_csv(file_path)
                print(f"  {file}: {len(df):,} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f" {file}: could not read ({e})")

if __name__ == "__main__":
    for domain in os.listdir(ROOT):
        sub_path = os.path.join(ROOT, domain)
        if os.path.isdir(sub_path):
            summarize_folder(sub_path)
