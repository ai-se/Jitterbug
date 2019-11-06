import pandas as pd
import os
from pdb import set_trace

for file in os.listdir("../data/"):
    with open("../data/"+file, "r") as f:
        data = f.read()
        set_trace()