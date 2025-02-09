# import libraries

from transformers import pipeline, Pipeline
import numpy as np
import pandas as pd
from typing import Callable
import os
import openpyxl





# function to import classifier from HF
def get_classifier() -> Pipeline:
    classifier = pipeline("zero-shot-classification")
    return classifier

def concat_sentences(df: pd.DataFrame) -> pd.DataFrame:
    sentence_count=1
    sentence = ""
    for i in range(len(df)):
        if i > 0 and df.loc[i, "speaker"] != df.loc[i-1, "speaker"]: 
            sentence = df.loc[i,"value"]
            df.loc[i,"sentence"] = sentence
            sentence_count+=1
        
        else: sentence = str(sentence) + " " + str(df.loc[i, "value"])
        df.loc[i,"sentence"] = sentence
        df.loc[i, "sentence_count"] = sentence_count
    df = df.groupby(["sentence_count"]).max().reset_index().drop(columns=["sentence_count","value"])
    return df



# read out data, can be modified to specify which data to read in
def get_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame()
    
    for file in os.listdir("Data"):
        temp = pd.read_csv("Data\\"+file, delimiter= "\t")
        temp = concat_sentences(temp)
        df = pd.concat([df,temp], axis=0, ignore_index= True)
       

    df_pat = df.loc[df["speaker"] == "Participant"].copy()
    df_ther = df.loc[df["speaker"] != "Participant"].copy()
    return df, df_pat, df_ther

# function to create the get_scores function, and automatically pass the classifier to it when it gets used.
def foobar(clfr: Pipeline) -> Callable[[pd.DataFrame], pd.Series]:
    def get_scores(col:pd.DataFrame) -> pd.Series:
        result = clfr(col, candidate_labels=['happy', 'unhappy', 'neutral'])
        return pd.Series({'sent': result["labels"][0], 'score': result["scores"][0]}) #type:ignore
    return get_scores

# function to use the get_scores function to classify the texts
def classify_texts(df: pd.DataFrame, get_scores) -> pd.DataFrame:
    df[['sent', 'score']] = df['sentence'].apply(get_scores)
    return df

# function to alter the sentiments if the confidence is below the treshold
def adjust_classifications(df: pd.DataFrame, treshold: float) -> pd.DataFrame:
    df["modified_sent"] = df["sent"].loc[df["score"]<=treshold].replace(['happy','unhappy'],'neutral')
    df["modified_sent"] = df["modified_sent"].fillna(df["sent"])
    return df

def main():
    classifier = get_classifier()
    df, df_pat, df_ther = get_data()
    df_pat =classify_texts(df_pat, foobar(classifier))
    df_pat = adjust_classifications(df_pat, 0.6)
    print(df_pat.to_markdown())
    df_pat.to_csv("first_results_grouping.csv", sep= ";")

if __name__ == "__main__":
    main()