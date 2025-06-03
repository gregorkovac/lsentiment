import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.LSenTiMent import LSenTiMent
from IPython.display import clear_output
import matplotlib.pyplot as plt
import argparse
import pickle
import json

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(X)

    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
def get_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader

def train(train_dataloader, val_dataloader, sentiment, lstm1_hidden_size, lstm1_num_layers, lstm2_hidden_size, lstm2_num_layers, window_size, lr, epochs, use_scheduler, scheduler_step, scheduler_gamma, device, mlp_layers, mlp_hidden_size, weight_decay, dropout_rate, double, plot_loss=True):
    np.random.seed(0)
    torch.manual_seed(0)

    input_size = 1

    model = LSenTiMent(input_size=input_size,
             hidden_size1=lstm1_hidden_size,
             num_layers1=lstm1_num_layers,
             hidden_size2=lstm2_hidden_size,
             num_layers2=lstm2_num_layers,
             window_size=window_size,
             sentiment=sentiment,
             dropout_rate=dropout_rate,
             mlp_hidden_size=mlp_hidden_size,
             mlp_layers=mlp_layers,
             double=double).to(device)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if use_scheduler:
        scheduler = StepLR(optimizer,
                           step_size=scheduler_step,
                           gamma=scheduler_gamma)

    best_val_loss = float('inf')
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train(True)

        train_loss = 0
        cnt = 0
        for i, batch in enumerate(train_dataloader):
            X, y = batch[0].to(device), batch[1].to(device)

            if not sentiment:
                X = X[:, :, 0:1]

            out = model(X)
            l = loss(out, y)
            train_loss += l.item()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            cnt += 1

        train_loss /= cnt

        train_losses.append(train_loss)

        model.train(False)

        val_loss = 0
        cnt = 0
        for i, batch in enumerate(val_dataloader):
            X, y = batch[0].to(device), batch[1].to(device)

            if not sentiment:
                X = X[:, :, 0:1]

            with torch.no_grad():
                out = model(X)
                l = loss(out, y)

                val_loss += l.item()

            cnt += 1

        val_loss /= cnt

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

            torch.save(model.state_dict(), "checkpoint.pt")

        val_losses.append(val_loss)

        if use_scheduler:
            scheduler.step()

        if epoch % 10 == 0:
            clear_output(wait=True)
            print(f"[ Epoch {epoch} / {epochs} ] Train loss = {train_loss} Val loss = {val_loss} Best val loss = {best_val_loss}")
            if plot_loss:
                plt.plot(train_losses, color="black", label="Train loss")
                plt.plot(val_losses, color="red", label="Validation loss")
                plt.yscale("log")
                plt.legend()
                plt.show()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best val loss: {best_val_loss}")

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",
                    help="Path to folder containing stocks CSV-s",
                    default=os.path.join(os.path.dirname(__file__), "../data/split"),
                    type=str)
    parser.add_argument("--model_path",
                        help="Model save directory",
                        default=os.path.join(os.path.dirname(__file__), "../models/"),
                        type=str)
    parser.add_argument("--use_sentiment",
                        help="Whether to use sentiment in the model or not",
                        action="store_true")
    parser.add_argument("--use_scheduler",
                        help="Whether to use learning rate scheduler or not",
                        action="store_true")

    parser.add_argument("--lstm1_hidden_size",
                        help="Hidden size of the LSTM",
                        default=64,
                        type=int)
    parser.add_argument("--lstm1_num_layers",
                        help="Number of layers of the LSTM",
                        default=1,
                        type=int)
    parser.add_argument("--lstm2_hidden_size",
                        help="Hidden size of the LSTM",
                        default=64,
                        type=int)
    parser.add_argument("--lstm2_num_layers",
                        help="Number of layers of the LSTM",
                        default=1,
                        type=int)
    parser.add_argument("--mlp_layers",
                        help="Number of layers of the MLP",
                        default=1,
                        type=int)
    parser.add_argument("--mlp_hidden_size",
                        help="Hidden layer size of the MLP",
                        default=256,
                        type=int)
    parser.add_argument("--window_size",
                        help="Size of the time series window",
                        default=5,
                        type=int)
    parser.add_argument("--lr",
                        help="Learning rate",
                        default=1e-5,
                        type=float)
    parser.add_argument("--epochs",
                        help="Number of epochs",
                        default=10000,
                        type=int)
    parser.add_argument("--scheduler_step",
                        help="Step of learning rate scheduler",
                        default=500,
                        type=int)
    parser.add_argument("--scheduler_gamma",
                        help="Gamma parameter of learning rate scheduler",
                        default=0.9,
                        type=float)
    parser.add_argument("--weight_decay",
                        help="Weight decay parameter",
                        default=0,
                        type=float)
    parser.add_argument("--dropout_rate",
                        help="Dropout rate parameter",
                        default=0,
                        type=float)
    parser.add_argument("--double",
                        help="Whether to use two LSTM-s when using sentiments",
                        action="store_true")
    parser.add_argument("--final",
                        help="Whether to also use the validation set for training",
                        action="store_true")

    
    args = parser.parse_args()

    data_path = args.data_path
    model_path = args.model_path
    use_sentiment = args.use_sentiment
    lstm1_hidden_size = args.lstm1_hidden_size
    lstm1_num_layers = args.lstm1_num_layers
    lstm2_hidden_size = args.lstm2_hidden_size
    lstm2_num_layers = args.lstm2_num_layers
    mlp_layers = args.mlp_layers
    mlp_hidden_size = args.mlp_hidden_size
    window_size = args.window_size
    lr = args.lr
    epochs = args.epochs
    use_scheduler = args.use_scheduler
    scheduler_step = args.scheduler_step
    scheduler_gamma = args.scheduler_gamma
    weight_decay = args.weight_decay
    dropout_rate = args.dropout_rate
    double = args.double
    final = args.final

    print(vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(data_path, "X_train.pkl"), "rb") as f:
        X_train = pickle.load(f)

    with open(os.path.join(data_path, "y_train.pkl"), "rb") as f:
        y_train = pickle.load(f)

    with open(os.path.join(data_path, "X_val.pkl"), "rb") as f:
        X_val = pickle.load(f)

    with open(os.path.join(data_path, "y_val.pkl"), "rb") as f:
        y_val = pickle.load(f)

    X_train = torch.cat(X_train)
    y_train = torch.cat(y_train)

    X_val = torch.cat(X_val)
    y_val = torch.cat(y_val)

    if final:
        X_train = torch.cat((X_train, X_val), dim=0)
        y_train = torch.cat((y_train, y_val), dim=0)

    train_dataloader, val_dataloader = get_dataloaders(X_train, y_train, X_val, y_val, batch_size=64)

    model = train(train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    sentiment=use_sentiment,
                    lstm1_hidden_size=lstm1_hidden_size,
                    lstm1_num_layers=lstm1_num_layers,
                    lstm2_hidden_size=lstm2_hidden_size,
                    lstm2_num_layers=lstm2_num_layers,
                    window_size=window_size,
                    lr=lr,
                    epochs=epochs,
                    use_scheduler=use_scheduler,
                    scheduler_step=scheduler_step,
                    scheduler_gamma=scheduler_gamma,
                    device=device,
                    mlp_layers=mlp_layers,
                    mlp_hidden_size=mlp_hidden_size,
                    weight_decay=weight_decay,
                    dropout_rate=dropout_rate,
                    double=double,
                    plot_loss=False)

    if use_sentiment:
        if double:
            model_file_path = os.path.join(model_path, "model_sentiment_double.pt")
            model_params_path = os.path.join(model_path, "model_sentiment_double_params.json")
        else:
            model_file_path = os.path.join(model_path, "model_sentiment_single.pt")
            model_params_path = os.path.join(model_path, "model_sentiment_single_params.json")
    else:
        model_file_path = os.path.join(model_path, "model_no_sentiment.pt")
        model_params_path = os.path.join(model_path, "model_no_sentiment_params.json")

    torch.save(model.state_dict(), model_file_path)
    with open(model_params_path, "w") as f:
        json.dump(vars(args), f)

    print(f"Saved model to {model_file_path}")


if __name__ == "__main__":
    main()