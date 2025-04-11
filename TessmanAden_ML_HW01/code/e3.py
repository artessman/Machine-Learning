
import pandas as pd
import numpy as np

#intialize df of lockers

myDict = {
    "Locker Num" : np.arange(1, 101),
    "Status" : np.zeros(100)
    }
#0 = closed
#1 = open
#all intiially closed
df = pd.DataFrame(myDict)

def flipBit(x):
    if x == 1:
        return 0
    else: 
        return 1


# loop to solve locker problem
for i in range(1, 101): 
    for j in range(100): 
        if (j + 1) % i == 0: 
            current_status = df.loc[j, "Status"]
            df.loc[j, "Status"] = flipBit(current_status)

open_lockers = df[df["Status"] == 1]
print("\nLockers that are open:")
print(open_lockers)
    