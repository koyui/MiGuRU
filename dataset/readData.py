# Author: koyui
# This File is designed to support chess dataset

import pandas as pd

def read_all():
    dataList = []

    transcriptList = []
    for i in range(2001, 2016):
        file_name = f"dataset/othello_csv_data/{i}.csv" # This path only called by training.py in the main directory
        df = pd.read_csv(file_name)
        transcript = df['transcript'].tolist()
        transcriptList += transcript

    for data in transcriptList:
        actionList = []
        characters = list(data)
        for i in range(len(data)):
            if i % 2 == 0:
                newAct = (int(characters[i + 1]) - 1, ord(characters[i]) - ord('a'))
                actionList.append(newAct)
                assert 0 <= newAct[0] < 8
                assert 0 <= newAct[1] < 8
        dataList.append(actionList)
    # print(dataList)
    # print(len(dataList))
    return dataList

read_all()


