import csv
import pandas as pd
import numpy as np
"""
Facilities, Clients, LP Relaxation, Multicommodity-Flow Based LP, Actual Optimal, Local Search
"""
df = pd.read_csv('results_22.csv')
# CFL-LP/OPT,Local Search/OPT,MFN-LP/OPT
for i in range(6):
    idx = i * 100 + 2
    mean1 = np.exp(np.log(df["CFL-LP/OPT"][idx:idx + 100]).mean())
    std1 = df["CFL-LP/OPT"][idx:idx + 100].std()

    mean3 = np.exp(np.log(df["Local Search/OPT"][idx:idx + 100]).mean())
    std3 = df["Local Search/OPT"][idx:idx + 100].std()

    if i < 5:
        mean2 = np.exp(np.log(df["MFN-LP/OPT"][idx:idx + 100]).mean())
        std2 = df["MFN-LP/OPT"][idx:idx + 100].std()

        print(df["Facilities"][idx], df["Clients"][idx],
            f"${mean1:.3f} \pm {std1:.3f}$ &", 
            f"${mean2:.3f} \pm {std2:.3f}$ &", 
            f"${mean3:.3f} \pm {std3:.3f}$")
    else:
        print(df["Facilities"][idx], df["Clients"][idx],
            f"${mean1:.3f} \pm {std1:.3f}$ &", 
            f"$-$ &", 
            f"${mean3:.3f} \pm {std3:.3f}$")