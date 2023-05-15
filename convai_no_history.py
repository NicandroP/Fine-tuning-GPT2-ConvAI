from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
from torch.utils.data import Dataset
import json
import pandas as pd
import numpy as np
import torch

class ChatData(Dataset):
    def __init__(self, path:str, tokenizer):
        self.data = json.load(open(path, "r"))

        self.X = []
        for i in self.data:
            lista=[]
            for j in i['dialog']:
                lista.append(j['text'])
            if(len(lista)>=10):
                self.X.append(lista)
        
      
        self.temp=[]
        for row in self.X:
            conv = list([x +" "+tokenizer.eos_token for x in row])
            conv = ' '.join(conv)
            self.temp.append(conv)
        self.X=self.temp
      

        self.X_encoded = tokenizer(self.X,max_length=80, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

def train(chatData,model,optim):
    epochs=500
    for i in tqdm.tqdm(range(epochs)):
        for X,a in chatData:
            X=X.to(device)
            a=a.to(device)
            optim.zero_grad()
            loss=model(X,attention_mask=a,labels=X).loss
            loss.backward()
            optim.step()
        torch.save(model.state_dict(),"model_state.pt")

tokenizer=GPT2Tokenizer.from_pretrained("gpt2")


tokenizer.add_special_tokens({"pad_token":"<pad>","bos_token":"<startofstring>","eos_token":"<endofstring>"})

model=GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

model = model.to(device)

chatData=ChatData("chat_data.json",tokenizer)
chatData=DataLoader(chatData,batch_size=64)

model.train()
optim=Adam(model.parameters(),lr=1e-3)

train(chatData,model,optim)

def infer(inp):
    inp=inp+" "+tokenizer.eos_token
    history.append(inp)
    print("history: ",history)
    join_history=' '.join(history)
    print("join_history: ",join_history)
    inp=tokenizer(join_history,max_length=80, truncation=True, padding="max_length", return_tensors="pt")
    print("inp tokenized: ",inp)


    X=inp["input_ids"].to(device)
    a=inp["attention_mask"].to(device)
    output=model.generate(X,attention_mask=a)

    output=tokenizer.decode(output[0])
    print("OUTPUT: ",output)
    output_no_pad = output.replace("<pad>", "").replace(" ", "").replace(",", "").replace("<endofstring>", " <endofstring>")
    history.append(output_no_pad)
    print(history)
    return output_no_pad.replace(" <endofstring>","")

history = []
while True:
  inp=input()
  print("OUTPUT: ",infer(inp))