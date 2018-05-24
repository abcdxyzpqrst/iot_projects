import pandas as pd
import csv
import pickle as pk

csv_file = pd.read_csv("raw/30_Industry_Portfolios_Daily.CSV", skiprows=9,
        low_memory=False)
csv_file = csv_file[:24200]
#csv_file = csv_file[csv_file.columns[1:]]
print(csv_file.describe())

with open("processed/portfolio.df", "wb") as file:
    pk.dump(csv_file, file)

csv_file.to_csv("processed/portfolio.csv")
