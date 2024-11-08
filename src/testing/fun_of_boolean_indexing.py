

import pandas as pd


dic = {
    "a": [1,1,1,0,0,1,1],
    "b": [0,0,1,1,0,0,0],
    "c": [1,0,1,0,0,1,0],
    "d": [0,0,1,0,0,1,0],
}

names = ["b","c","d"]


df = pd.DataFrame(data=dic)

print("df")
print(df)
print("")

df_A = df.loc[~(df[names] == 0.0).all(axis=1)]


print("df_A = df.loc[~(df[names] == 0.0).all(axis=1)]")
print(df_A)
print("")

df_B = df.loc[(df[names] != 0.0).all(axis=1)]

print("df_B = df.loc[(df[names] != 0.0).all(axis=1)]")
print(df_B)
print("")


df_C = df.loc[(df[names] != 0.0).any(axis=1)]

print("df_C = df.loc[(df[names] != 0.0).any(axis=1)]")
print(df_C)
print("")

df_D = df.loc[~(df[names] == 0.0).any(axis=1)]

print("df_D = df.loc[~(df[names] == 0.0).any(axis=1)]")
print(df_D)
print("")