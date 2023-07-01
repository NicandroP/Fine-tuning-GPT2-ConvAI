import torch
from torch.jit import script, trace
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import string


import re
import json
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)
def printLines(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

#printLines("utterances.jsonl")

# Splits each line of the file to create lines and conversations
def loadLinesAndConversations(fileName):
    lines = {}
    conversations = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            lineJson = json.loads(line)
            # Extract fields for line object
            lineObj = {}
            lineObj["lineID"] = lineJson["id"]
            lineObj["characterID"] = lineJson["speaker"]
            lineObj["text"] = lineJson["text"]
            lines[lineObj['lineID']] = lineObj

            # Extract fields for conversation object
            if lineJson["conversation_id"] not in conversations:
                convObj = {}
                convObj["conversationID"] = lineJson["conversation_id"]
                convObj["movieID"] = lineJson["meta"]["movie_id"]
                convObj["lines"] = [lineObj]
            else:
                convObj = conversations[lineJson["conversation_id"]]
                convObj["lines"].insert(0, lineObj)
            conversations[convObj["conversationID"]] = convObj

    return lines, conversations

def clean_string(input_string):
    allowed_chars = string.ascii_letters + string.digits + string.punctuation + " "
    cleaned_string = "".join(char for char in input_string if char in allowed_chars)
    return cleaned_string

# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations,len1):
    qa_pairs = []
    for conversation in conversations.values():
        single_conversation=[]
        # Iterate over all the lines of the conversation
        if len(conversation["lines"])>=len1:
            for i in range(len1):  # We ignore the last line (no answer for it)
                text=conversation["lines"][i]["text"].strip()
                cleaned_text=clean_string(text)
                cleaned_text=text.replace('"', '')
                single_conversation.append(cleaned_text)
        if single_conversation:
                qa_pairs.append(single_conversation)
        
    return qa_pairs

# Define path to new file
datafile = "formatted_movie_lines.txt"

delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# Initialize lines dict and conversations dict
lines = {}
conversations = {}
# Load lines and conversations
print("\nProcessing corpus into lines and conversations...")
lines, conversations = loadLinesAndConversations("utterances.jsonl")
dataset=extractSentencePairs(conversations,4)
print(dataset)
print(len(dataset))
"""# Write new csv file
print("\nWriting newly formatted file...")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations,6):
        writer.writerow(pair)

# Print a sample of lines
print("\nSample lines from file:")
printLines(datafile)"""
import pandas as pd
df = pd.DataFrame(dataset, columns=["context/3","context/2","context/1", "context/0", "context", "response"])
df = df.iloc[:, ::-1]
df.to_csv("dataset6.csv", index=False)