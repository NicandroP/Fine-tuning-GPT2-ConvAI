import pandas as pd
import json
import csv
df = json.load(open("chat_data.json", "r"))

X=[]
for i in df:
    lista=[]
    for j in i['dialog']:
        lista.append(j['text'])
    if(len(lista)>=6):
        X.append(lista)

print(X)
print(len(X))
"""for i in range(len(X)):
    X[i]=X[i][:12]
    X[i]=X[i][::-1]

columns=['response', 'context', 'context/0', 'context/1', 'context/2',
       'context/3', 'context/4', 'context/5', 'context/6', 'context/7',
       'context/8', 'context/9']

df = pd.DataFrame(X, columns=columns)"""




"""import pandas as pd

df = pd.read_csv('casual_data_windows.csv')

df=df.drop("Unnamed: 0", axis=1)
df = df.iloc[:, ::-1]
df.columns = ['response', 'context', 'context/0']
print(df)
print(df.columns)

df.to_csv('reddit_3_conversation.csv', index=False)
"""