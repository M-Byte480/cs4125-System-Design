import numpy as np
import pandas as pd

df = pd.read_csv("AppGallery.csv")

# convert the dtype object to unicode string
df['Interaction content'] = df['Interaction content'].values.astype('U')
df['Ticket Summary'] = df['Ticket Summary'].values.astype('U')

#Optional: rename variable names for remebering easily
df["y1"] = df["Type 1"]
df["y2"] = df["Type 2"]
df["y3"] = df["Type 3"]
df["y4"] = df["Type 4"]
df["x"] = df['Interaction content']

df["y"] = df["y2"]

# remove empty y
df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
df.shape

print(df.shape)
