import numpy as np
import pandas as pd

# Data pre-processing
df = pd.read_csv("Live.csv")
pd.set_option('display.max_columns', None)
df.head()
df = df.drop(["Column1", "Column2", "Column3", "Column4"], axis=1)
df.shape
