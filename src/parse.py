import pandas as pd
import os
from pdb import set_trace

for file in os.listdir("../data/"):
    df = pd.read_csv("../data/"+file)
    df.rename(columns={'commenttext':'Abstract'}, inplace=True)
    df['code'] = ["no" if type=="WITHOUT_CLASSIFICATION" else "yes" for type in df["classification"]]
    df.to_csv("../new_data/"+file, line_terminator="\r\n")
