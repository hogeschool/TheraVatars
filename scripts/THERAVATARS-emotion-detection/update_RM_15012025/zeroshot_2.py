# import libraries
from transformers import pipeline, Pipeline
import numpy as np
import pandas as pd
from typing import Callable
import os


# function to import classifier from HF
def get_classifier() -> Pipeline:
    classifier = pipeline("zero-shot-classification")
    return classifier

def concat_sentences(df: pd.DataFrame) -> pd.DataFrame:
    sentence_count=1
    sentence = ""
    for i in range(len(df)):
        if i > 0 and df.loc[i, "speaker"] != df.loc[i-1, "speaker"] and float(df.loc[i, "stop_time"]) - float(df.loc[i-1,"start_time"]) >= 0.5: #type:ignore
            sentence = df.loc[i,"value"]
            df.loc[i,"sentence"] = sentence
            sentence_count+=1
        
        else: sentence = str(sentence) + " " + str(df.loc[i, "value"])
        df.loc[i,"sentence"] = sentence
        df.loc[i, "sentence_count"] = int(sentence_count)
    df = df.loc[df.groupby('sentence_count')['sentence'].apply(lambda x: x.str.len().idxmax())].reset_index(drop=True)
    df = df.drop(columns = ["value", "sentence_count"])
        
    return df



# read out data, can be modified to specify which data to read in
def get_data(dirname) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # for multiple files:
    df = pd.DataFrame()
    for file in os.listdir(dirname):
        temp = pd.read_csv(f"{dirname}/"+file, delimiter= "\t")
        temp = concat_sentences(temp)
        #temp["sentence"] = temp["sentence"].apply(lambda x: ' '.join(x))
        df = pd.concat([df,temp], axis=0, ignore_index= True)
    df_pat = df.loc[df["speaker"] == "Participant"]
    df_ther = df.loc[df["speaker"] != "Participant"]

    return df, df_pat, df_ther

# function to create the get_scores function, and automatically pass the classifier to it when it gets used.
def get_scores_func_creator(clfr: Pipeline) -> Callable[[pd.DataFrame], pd.Series]:
    def get_scores(col:pd.DataFrame) -> pd.Series:
        result = clfr(col, candidate_labels=['happy', 'unhappy', 'neutral'])
        return pd.Series({'sent': result["labels"][0], 'score': result["scores"][0]}) #type:ignore
    return get_scores

# function to use the get_scores function to classify the texts
def classify_texts(df: pd.DataFrame, get_scores: Callable, amount: int) -> pd.DataFrame:
    df = df.loc[df["sentence"] != ''][:amount]
    df[['sent', 'score']] = df['sentence'].apply(get_scores)

    return df

# function to alter the sentiments if the confidence is below the treshold
def adjust_classifications(df: pd.DataFrame, treshold: float) -> pd.DataFrame:
    df["modified_sent"] = df["sent"].loc[df["score"]<=treshold].replace(['happy','unhappy'],'neutral')
    df["modified_sent"] = df["modified_sent"].fillna(df["sent"])
    return df

def main():
    classifier = get_classifier()
    print("one")
    df, df_pat, df_ther = get_data('../root/workspace/THERAVATARS_NN/transcripts')
    print("two")
    df_pat =classify_texts(df_pat, get_scores_func_creator(classifier), 1000)
    print("three")
    df_pat = adjust_classifications(df_pat, 0.6)
    print("four")
    df_pat.to_csv("/root/workspace/THERAVATARS_RM/resultaat_2.csv", sep= ";")

if __name__ == "__main__":
    main()