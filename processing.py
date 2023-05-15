import pandas as pd
import json
df = json.load(open("chat_data.json", "r"))

X=[]
for i in df:
    lista=[]
    for j in i['dialog']:
        lista.append(j['text'])
    if(len(lista)>=12):
        X.append(lista)

for i in range(len(X)):
    X[i]=X[i][:12]
    X[i]=X[i][::-1]

columns=['response', 'context', 'context/0', 'context/1', 'context/2',
       'context/3', 'context/4', 'context/5', 'context/6', 'context/7',
       'context/8', 'context/9']

df = pd.DataFrame(X, columns=columns)

print(df)

df.to_csv('en_conv_small.csv', index=False)