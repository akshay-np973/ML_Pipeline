from email import header

import pandas as pd
import sys
import os
import yaml

params = yaml.safe_load(open("params.yaml"))["preprocess"]

def preprocess_data(data_path, output_path):
    # Load the data
    df = pd.read_csv(data_path)

    #preprocess the data and EDA
    df = df.dropna()
      # Drop rows with missing values
    df = df.drop_duplicates()  
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path,header=None, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    data_path = params["input"]
    output_path = params["output"]
    preprocess_data(data_path, output_path)



    