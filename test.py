import pandas as pd
from datasets import load_dataset

df=load_dataset("danidanou/Bloomberg_Financial_News")


print(df)
print("nom des splits",df.keys())
train_df = df['train'].to_pandas()
print(train_df)
print("structure des features",train_df.columns)

print(df['train'][1])