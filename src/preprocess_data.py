import pandas as pd
import os
from datetime import datetime, date
from tqdm import tqdm
import re
import numpy as np
import torch

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

def apply_time_window(df, col_name, n):
    df = df.copy()
    for i in range(n, 0, -1):
        df[f"{col_name}-{i}"] = df[col_name].shift(i)

    df.dropna(inplace=True)

    return df

# def prepare_stocks_date(file_path, date_column_name="Date", focus_price_column_name="Close"):
#     df_stocks = pd.read_csv(file_path)

#     df_stocks = df_stocks[[date_column_name, focus_price_column_name]]

#     stock_mean = df_stocks[focus_price_column_name].mean()
#     stock_std = df_stocks[focus_price_column_name].std()

#     df_stocks[focus_price_column_name] = (df_stocks[focus_price_column_name] - stock_mean) / stock_std

#     df_stocks["Date"] = df_stocks["Date"].apply(lambda x: x.split(" ")[0])
#     df_stocks["Date"] = df_stocks["Date"].apply(convert_date)

#     df_stocks.set_index("Date", inplace=True)

#     df_stocks = apply_time_window(df = df_stocks,
#                                                   price_col_name=FOCUS_PRICE,
#                                                   n = WINDOW_SIZE)

#     return df_stocks, stock_mean, stock_std

def preprocess_data(stock_path, sentiment_path, stocks_date_col="date", stocks_price_col="close", sent_date_col="timestamp", sent_sent_col="sentiment"):
    # Prepare stocks data
    df_stocks = pd.read_csv(stock_path)

    df_stocks = df_stocks[[stocks_date_col, stocks_price_col]]

    df_stocks[stocks_date_col] = df_stocks[stocks_date_col].apply(lambda x: x.split(" ")[0])
    df_stocks[stocks_date_col] = df_stocks[stocks_date_col].apply(convert_date)

    df_stocks.set_index(stocks_date_col, inplace=True)

    # Prepare sentiment data
    df_sentiment = pd.read_csv(sentiment_path)

    df_sentiment = df_sentiment[[sent_date_col, sent_sent_col]]

    df_sentiment.columns = [sent_date_col, sent_sent_col]

    df_sentiment[sent_date_col] = pd.to_datetime(df_sentiment[sent_date_col])
    df_sentiment.set_index(sent_date_col, inplace=True)

    # Merge data
    df = pd.merge(df_stocks, df_sentiment, how="left", left_index=True, right_index=True)

    # Drop rows with NA sentiment
    df = df[df[sent_sent_col].notna()]

    df.columns = ["price", "sentiment"]

    return df

def apply_time_window_both(df, window_size):
    df = apply_time_window(df, col_name="price", n=window_size)
    df = apply_time_window(df, col_name="sentiment", n=window_size)

    df = df[df["price"].notna()]
    df = df[df["sentiment"].notna()]

    return df

def df_to_torch(df, window_size):
    Xy = df.to_numpy()
    
    X = Xy[:, 2:]
    X = np.stack((X[:, :window_size], X[:, window_size:]), axis=2)
    X = torch.tensor(X, dtype=torch.float32)

    y = Xy[:, :1]
    y = torch.tensor(y, dtype=torch.float32)

    return X, y

def split_and_format_data(df, window_size=3, split1=0.7, split2=0.9):
    split1_ = int(len(df) * split1)
    split2_ = int(len(df) * split2)

    df_train = df.iloc[:split1_]
    df_val = df.iloc[split1_:split2_]
    df_test = df.iloc[split2_:]

    mean = df_train["price"].mean()
    std = df_train["price"].std()

    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    df_train["price"] = (df_train["price"] - mean) / std
    df_val["price"] = (df_val["price"] - mean) / std
    df_test["price"] = (df_test["price"] - mean) / std

    df_train = apply_time_window_both(df=df_train, window_size=window_size)
    df_val = apply_time_window_both(df=df_val, window_size=window_size)
    df_test = apply_time_window_both(df=df_test, window_size=window_size)

    X_train, y_train = df_to_torch(df_train, window_size)
    X_val, y_val = df_to_torch(df_val, window_size)
    X_test, y_test = df_to_torch(df_test, window_size)

    return X_train, y_train, X_val, y_val, X_test, y_test, mean, std
