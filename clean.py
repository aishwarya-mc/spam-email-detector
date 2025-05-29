import pandas as pd

df=pd.read_csv("data/spam.csv",encoding='latin-1')
df=df[['v1','v2']]
df.columns=['label','message']

df.to_csv('data/cleaned_spam.csv',index=False)

print(df.head())
print(df['label'].value_counts())