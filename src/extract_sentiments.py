import pandas as pd
import os
import sys
from tqdm import tqdm
sys.path.insert(0, "..")
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', 'finBERT')))
import finBERT.finbert.finbert as finbert
from transformers import AutoModelForSequenceClassification
import logging

# Disable logging from FinBERT
logging.getLogger().setLevel(logging.ERROR)

def get_sentiment(text, model):
    # Get predictions from FinBERT
    result = finbert.predict(text, model)

    # Return mean sentiment
    return result["sentiment_score"].mean()

def get_sentiment_for_df(df):

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert",num_labels=3,cache_dir=None)

    # Get average sentiment for each day
    sentiments = []
    for text in tqdm(df["description"]):
        sentiments.append(get_sentiment(text, model))

    df["sentiments"] = sentiments

    return df

def main():

    args = sys.argv
    if len(args) == 1:
        print("Usage: python extract_sentiment.py <data_path> <timestamp_name> <text_name>")
        print("\t\t <data_path> - path to CSV file with texts")
        print("\t\t <timestamp_name> - name of the timestamp column in the CSV")
        print("\t\t <text_name> - name of the text column in the CSV")
        exit(0)

    data_path = args[1]
    if len(args) > 2:
        timestamp_name = args[2]
    else:
        timestamp_name = "timestamp"

    if len(args) > 3:
        text_name = args[3]
    else:
        text_name = "description"

    print("Reading data...")

    # Read the CSV and keep only used columns
    df = pd.read_csv(data_path)
    df = df[[timestamp_name, text_name]]
    
    # Convert timestamp to date
    df[timestamp_name] = pd.to_datetime(df[timestamp_name], format='ISO8601')
    df[timestamp_name] = df[timestamp_name].dt.date

    # Drop NaN values
    df = df.dropna(subset=[text_name])

    # Join texts by date
    grouped_df = df.groupby(timestamp_name)[text_name].apply(lambda x: ' '.join(x)).reset_index()

    # Extract sentiment
    grouped_df = get_sentiment_for_df(grouped_df)

    # Save the result
    out_path = data_path.split(".csv")[0] + "_sentiments.csv"
    grouped_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()