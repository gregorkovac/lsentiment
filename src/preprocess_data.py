import pandas as pd
import os
from datetime import datetime, date
from tqdm import tqdm
import re
import numpy as np
import torch
import pickle
import argparse
import json

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

    # df_sentiment = df_sentiment[[sent_date_col, sent_sent_col]]
    df_sentiment = df_sentiment[[sent_date_col, "neg", "neu", "pos"]]


    # df_sentiment.columns = [sent_date_col, sent_sent_col]
    df_sentiment.columns = [sent_date_col, "neg", "neu", "pos"]

    df_sentiment[sent_date_col] = pd.to_datetime(df_sentiment[sent_date_col])
    df_sentiment.set_index(sent_date_col, inplace=True)

    # Merge data
    df = pd.merge(df_stocks, df_sentiment, how="left", left_index=True, right_index=True)

    # Drop rows with NA sentiment
    # df = df[df[sent_sent_col].notna()]
    df = df[df["neg"].notna()]
    df = df[df["neu"].notna()]
    df = df[df["pos"].notna()]

    # df.columns = ["price", "sentiment"]
    df.columns = ["price", "neg", "neu", "pos"]

    return df

def apply_time_window_both(df, window_size):
    df = apply_time_window(df, col_name="price", n=window_size)
    df = apply_time_window(df, col_name="neg", n=window_size)
    df = apply_time_window(df, col_name="neu", n=window_size)
    df = apply_time_window(df, col_name="pos", n=window_size)

    df = df[df["price"].notna()]
    # df = df[df["sentiment"].notna()]

    df = df.drop(columns=["pos"])
    df = df.drop(columns=["neu"])
    df = df.drop(columns=["neg"])

    return df

def df_to_torch(df, window_size):

    Xy = df.to_numpy()

    X = Xy[:, 1:]
    X = np.stack((X[:, :window_size], 
                  X[:, window_size:2*window_size],
                  X[:, 2*window_size:3*window_size],
                  X[:, 3*window_size:]), axis=2)
    X = torch.tensor(X, dtype=torch.float32)

    y = Xy[:, :1]
    y = torch.tensor(y, dtype=torch.float32)

    return X, y

def split_data(df, split1, split2):
    split1_ = int(len(df) * split1)
    split2_ = int(len(df) * split2)

    df_train = df.iloc[:split1_]
    df_val = df.iloc[split1_:split2_]
    df_test = df.iloc[split2_:]

    return df_train, df_val, df_test

def format_data(df_train, df_val, df_test, price_mean, price_std, sentiment_mean, sentiment_std, window_size=3):
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    df_train["price"] = (df_train["price"] - price_mean) / price_std
    df_val["price"] = (df_val["price"] - price_mean) / price_std
    df_test["price"] = (df_test["price"] - price_mean) / price_std

    # df_train["sentiment"] = (df_train["sentiment"] - sentiment_mean) / sentiment_std
    # df_val["sentiment"] = (df_val["sentiment"] - sentiment_mean) / sentiment_std
    # df_test["sentiment"] = (df_test["sentiment"] - sentiment_mean) / sentiment_std

    df_train = apply_time_window_both(df=df_train, window_size=window_size)
    df_val = apply_time_window_both(df=df_val, window_size=window_size)
    df_test = apply_time_window_both(df=df_test, window_size=window_size)

    X_train, y_train = df_to_torch(df_train, window_size)
    X_val, y_val = df_to_torch(df_val, window_size)
    X_test, y_test = df_to_torch(df_test, window_size)

    return X_train, y_train, X_val, y_val, X_test, y_test

def process_and_save_data(stock_paths, sentiment_path, window_size, out_path):
    df_train = []
    df_val = []
    df_test = []
    
    train_prices = []
    train_sentiments = []
    
    for stock_path in tqdm(stock_paths):
        stock_name = stock_path.split("/")[-1].split(".csv")[0]
    
        if stock_name in ["netflix", "meta"]:
            date_col = "Date"
            price_col = "Close"
        else:
            date_col = "date"
            price_col = "close"
    
        df = preprocess_data(stock_path, sentiment_path, stocks_date_col=date_col, stocks_price_col=price_col)
        df_train_, df_val_, df_test_ = split_data(df, split1=0.7, split2=0.9)
        
        df_train.append(df_train_)
        df_val.append(df_val_)
        df_test.append(df_test_)
    
        train_prices.append(df_train_["price"].values)
        # train_sentiments.append(df_train_["sentiment"].values)
        # train_sentiments.append(df_train_["pos"].values)
        # train_sentiments.append(df_train_["neu"].values)
        # train_sentiments.append(df_train_["neg"].values)
    
    price_mean = np.mean(np.concatenate(train_prices))
    price_std = np.std(np.concatenate(train_prices))
    
    # sentiment_mean = np.mean(np.concatenate(train_sentiments))
    # sentiment_std = np.std(np.concatenate(train_sentiments))

    sentiment_mean = 0
    sentiment_std = 1
    
    X_train, y_train = [], []
    X_val, y_val = [], []
    
    Xy_val = []
    Xy_test = []
    
    for i in range(len(df_train)):
        X_train_, y_train_, X_val_, y_val_, X_test_, y_test_ = format_data(df_train[i], df_val[i], df_test[i], price_mean, price_std, sentiment_mean, sentiment_std, window_size)
    
        X_train.append(X_train_)
        y_train.append(y_train_)
    
        X_val.append(X_val_)
        y_val.append(y_val_)
    
        Xy_test.append((X_test_, y_test_))
        Xy_val.append((X_val_, y_val_))


    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with open(os.path.join(out_path, "X_train.pkl"), "wb") as f:
        pickle.dump(X_train, f)

    with open(os.path.join(out_path, "y_train.pkl"), "wb") as f:
        pickle.dump(y_train, f)

    with open(os.path.join(out_path, "X_val.pkl"), "wb") as f:
        pickle.dump(X_val, f)

    with open(os.path.join(out_path, "y_val.pkl"), "wb") as f:
        pickle.dump(y_val, f)

    with open(os.path.join(out_path, "Xy_test.pkl"), "wb") as f:
        pickle.dump(Xy_test, f)

    with open(os.path.join(out_path, "norms.json"), "w") as f:
        json.dump({
            "price_mean": price_mean,
            "price_std": price_std,
            "sentiment_mean": sentiment_mean,
            "sentiment_std": sentiment_std
        }, f)

    print(f"Saved split data to {out_path}")

    

    # X_train = torch.cat(X_train)
    # y_train = torch.cat(y_train)
    
    # X_val = torch.cat(X_val)
    # y_val = torch.cat(y_val)
    
    # # X_train, y_train, X_val, y_val, X_test, y_test, mean, std = split_and_format_data(df, window_size=WINDOW_SIZE, split1=0.7, split2=0.9)
    # train_dataloader, val_dataloader = get_dataloaders(X_train, y_train, X_val, y_val, batch_size=64)
    
    # print(f"Train set length = {len(X_train)}")
    # print(f"Val set length = {len(X_val)}")
    # print(f"Test set length (per stock) = {len(Xy_test[0][0])}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stocks_path",
                        help="Path to folder containing stocks CSV-s",
                        default=os.path.join(os.path.dirname(__file__), "../data/maang_stocks"))
    parser.add_argument("--sentiment_path",
                        help="Path to sentiment CSV.",
                        default=os.path.join(os.path.dirname(__file__), "../data/financial_tweets_sentiments.csv"))
    parser.add_argument("--out_path",
                        help="Output folder path",
                        default=os.path.join(os.path.dirname(__file__), "../data/split"))
    parser.add_argument("--window_size",
                        help="Size of the time series window",
                        default=10)
    
    args = parser.parse_args()

    stocks_path = args.stocks_path
    sentiment_path = args.sentiment_path
    out_path = args.out_path
    window_size = args.window_size

    stock_paths = []
    for s in os.listdir(stocks_path):
        if ".csv" in s:
            stock_paths.append(os.path.join(stocks_path, s))

    process_and_save_data(stock_paths=stock_paths,
                          sentiment_path=sentiment_path,
                          out_path=out_path,
                          window_size=window_size)
    
if __name__ == "__main__":
    main()