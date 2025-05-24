import pandas as pd
import os
import sys
from tqdm import tqdm
sys.path.insert(0, "..")
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'finBERT')))
import finBERT.finbert.finbert as finbert
from transformers import AutoModelForSequenceClassification
import logging

logging.getLogger().setLevel(logging.ERROR)

def get_sentiment(text, model):
    # with suppress_output():
    result = finbert.predict(text, model)

    return result["sentiment_score"].mean()

def get_sentiment_for_df(df):
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert",num_labels=3,cache_dir=None)

    # tqdm.pandas(desc="Analyzing sentiment")

    sentiments = []
    for text in tqdm(df["description"]):
        sentiments.append(get_sentiment(text, model))

    # df["sentiment"] = df["description"].progress_apply(lambda x: get_sentiment(x, model))
    df["sentiments"] = sentiments

    return df

def main():
    print("Reading data...")

    df = pd.read_csv("../data/financial_tweets/financial_tweets.csv")
    df = df[["timestamp", "description"]]
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    df['timestamp'] = df['timestamp'].dt.date

    df = df.dropna(subset=['description'])

    grouped_df = df.groupby('timestamp')['description'].apply(lambda x: ' '.join(x)).reset_index()

    grouped_df = get_sentiment_for_df(grouped_df)

    grouped_df.to_csv("../data/financial_tweets_sentiments.csv", index=False)

if __name__ == "__main__":
    main()