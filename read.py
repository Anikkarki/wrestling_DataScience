import pandas as pd
df=pd.read_csv('WWE_History_1000.csv')
print(df)
print(df["Winner"].repeat())