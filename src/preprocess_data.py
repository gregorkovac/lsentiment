import pandas as pd
import os
from datetime import datetime, date
from tqdm import tqdm
import re

############################ SENTIMENT140 ############################

def read_sentiment140(data_path):
    df = pd.read_csv(data_path, encoding='latin1', header=None)

    # Manually add columns since they are not present in the original csv
    df.columns = ["target", "ids", "date", "flag", "user", "text"]

    return df

def introduce_neutral_sentiment(df):
    # Extract ids value counts
    vc = df["ids"].value_counts()

    # Find duplicate ids
    double_ids = vc[vc != 1].index

    # Set rows with duplicate ids to neutral sentiment
    df.loc[df["ids"].isin(double_ids), "target"] = 2

    # Drop duplicates
    df = df.drop_duplicates(subset="ids", keep="first")

    return df

def format_date(df):
    date_clean = df['date'].str.replace('PDT', '', regex=False).str.strip()
    date_clean = pd.to_datetime(date_clean)
    df["date"] = date_clean

    return df

def clean_tweet(text):
    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags
    text = re.sub(r"#\w+", "", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Remove numbers
    # text = re.sub(r"\d+", "", text) 

    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_text(df):
    df["text"] = df["text"].apply(clean_tweet)

    return df


######################## STOCK MARKET DATASET ########################

def read_stocks(data_path, min_date = datetime(2009, 4, 6), max_date = datetime(2009, 6, 25)):
    path_etfs = os.path.join(data_path, "etfs")
    path_stocks = os.path.join(data_path, "stocks")

    dir_etfs = os.listdir(path_etfs)
    dir_stocks = os.listdir(path_stocks)

    etfs_list = []
    stocks_list = []

    for etf in tqdm(dir_etfs):
        df_etf = pd.read_csv(os.path.join(path_etfs, etf), parse_dates=["Date"])

        # if pd.to_datetime(df_etf["Date"].max()) < min_date or pd.to_datetime(df_etf["Date"].min()) > max_date:
        #     continue

        # df_etf = df_etf[(df_etf["Date"] >= min_date) & (df_etf["Date"] <= max_date)].copy()

        df_etf["Name"] = etf.split(".")[0]

        etfs_list.append(df_etf)

    for stock in tqdm(dir_stocks):
        df_stock = pd.read_csv(os.path.join(path_stocks, stock), parse_dates=["Date"])

        # if pd.to_datetime(df_stock["Date"].max()) < min_date or pd.to_datetime(df_stock["Date"].min()) > max_date:
        #     continue

        # df_stock = df_stock[(df_stock["Date"] >= min_date) & (df_stock["Date"] <= max_date)].copy()

        df_stock["Name"] = stock.split(".")[0]

        stocks_list.append(df_stock)

    df_etfs = pd.concat(etfs_list, ignore_index=True)
    df_stocks = pd.concat(stocks_list, ignore_index=True)

    return df_etfs, df_stocks

def remove_rare_points(df):
    return df[df.groupby("Name")["Name"].transform("count") >= 10]

def one_hot_encode_name(df):
    pass

def convert_date(d):
    year, month, day = d.split("-")

    year = int(year)
    month = int(month)
    day = int(day)

    return date(year=year, month=month, day=day)

def apply_time_window(df, price_col_name, n):
    for i in range(n, 0, -1):
        df[f"{price_col_name}-{i}"] = df[price_col_name].shift(i)

    df.dropna(inplace=True)

    return df
