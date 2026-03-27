import pandas as pd

df = pd.read_csv("../outputs/evaluation_table.csv")

pd.set_option('display.max_colwidth', None)
print(df.head())