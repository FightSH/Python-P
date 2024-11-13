import pandas as pd
import numpy as np

df = pd.DataFrame({'key1':['a','a','b','d','a'],
                  'key2':['one','two','one','two','one'],
                  'data1':[1,3,3,4,5],
                  'data2':np.random.randint(5)})

print(df)

# 均值
mean = df['data1'].groupby(df['key1']).mean()
print(mean)

mean2 = df['data1'].groupby(df['key1'], df['key2']).mean()
print(mean2)
