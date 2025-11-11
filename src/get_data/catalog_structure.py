import os
import pandas as pd

# root folder 
ROOT = "data/raw/FakeNewsNet"

# function to summarize the CSV files
def summarize_folder(path):
    # print the name of the folder that is being summarized
    print(f"\n Folder: {path}")
    # loop through the files in the folder
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file.endswith(".csv"):

            #skip the empty lines
            if os.path.getsize(file_path) == 0:
                print (f" {file}: empty file, skipped")
                continue 
            try:
                # read the csv file
                df = pd.read_csv(file_path)
                print(f"  {file}: {len(df):,} rows, {len(df.columns)} columns")
            
            # throw error exception 
            except Exception as e:
                print(f" {file}: could not read ({e})")

if __name__ == "__main__":
    # loop through each item in the root directory
    for domain in os.listdir(ROOT):
        sub_path = os.path.join(ROOT, domain)
        # if path exists, call function to summarize CSV files
        if os.path.isdir(sub_path):
            summarize_folder(sub_path)
