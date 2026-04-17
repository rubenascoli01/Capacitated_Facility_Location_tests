import csv
import pandas as pd
import numpy as np
"""
Facilities, Clients, LP Relaxation, Multicommodity-Flow Based LP, Actual Optimal, Local Search
"""
df = pd.read_csv('results.csv')
# CFL-LP/OPT,Local Search/OPT,MFN-LP/OPT
for i in range(5):
    idx = i * 100 + 2
    mean1 = np.exp(np.log(df["MFN-LP/OPT"][idx:idx + 100]).mean())
    std1 = df["MFN-LP/OPT"][idx:idx + 100].std()

    print(df["Facilities"][idx], df["Clients"][idx], mean1, std1)