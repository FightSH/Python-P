import  pandas as pd

df = pd.read_json('./data/1.json')
print(df)

df.to_excel('./data/output.xlsx', index=False)
