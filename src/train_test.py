import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
sys.path.insert(0, "..")
from models.LSenTiMent import LSenTiMent
from IPython.display import clear_output
import matplotlib.pyplot as plt

STOCK_PATH = "../data/apple_stock/AAPL_1980-12-03_2025-03-15.csv"
SENTIMENT_PATH = "../data/financial_tweets/financial_tweets_sentiments.csv"

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(X)

    def __len__(self):
        return self.len
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
def get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def train(train_dataloader, val_dataloader, sentiment, lstm_hidden_size, lstm_num_layers, window_size, lr, epochs, use_scheduler, scheduler_step, scheduler_gamma, device, mlp_layers, mlp_hidden_size):
    np.random.seed(0)
    torch.manual_seed(0)

    input_size = 1

    model = LSenTiMent(input_size=input_size,
             hidden_size=lstm_hidden_size,
             num_layers=lstm_num_layers,
             window_size=window_size,
             sentiment=sentiment,
             mlp_hidden_size=mlp_hidden_size,
             mlp_layers=mlp_layers).to(device)

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        val_losses.append(val_loss)

        if use_scheduler:
            scheduler.step()

        if epoch % 100 == 0:
            clear_output(wait=True)
            print(f"[ Epoch {epoch} / {epochs} ] Train loss = {train_loss} Val loss = {val_loss} Best val loss = {best_val_loss}")
            plt.plot(train_losses, color="black", label="Train loss")
            plt.plot(val_losses, color="red", label="Validation loss")
            plt.yscale("log")
            plt.legend()
            plt.show()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best val loss: {best_val_loss}")

    return model


if __name__ == "__main__":
    main()