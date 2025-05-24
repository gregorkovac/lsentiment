import torch
from torch import nn

class LSenTiMent(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, window_size, sentiment = True):
        super().__init__()

        # Save parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.sentiment = sentiment

        # LSTM for stock price
        self.lstm_stock = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        
        # Fully-connected layer for stock price LSTM
        self.fc = nn.Linear(hidden_size, 1)

        # LSTM for sentiment
        self.lstm_sentiment = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # Final multi-layer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(window_size, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        # Extract batch size
        batch_size = x.size(0)

        # Initialize hidden and cell states for LSTM
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        if self.sentiment:
            # Extract stock prices in time window
            x_stock = x[:, :, 0:1]

            # Extract sentiments in time window
            x_sentiment = x[:, :, 1:2]

            # Obtain stock results from LSTM
            x_stock, _ = self.lstm_stock(x_stock, (h_0, c_0))
            
            # Obtain sentiment results from LSTM
            x_sentiment, _ = self.lstm_sentiment(x_sentiment, (h_0, c_0))

            # Extract last prediction
            x_stock = x_stock[:, -1, :]
            x_sentiment = x_sentiment[:, -1, :]

            # Pass stock price through the fully connected layer
            x_stock = self.fc(x_stock)

            # Concatenate predicted stock price and sentiment results from LSTM
            x = torch.cat((x_stock, x_sentiment), dim=1)

            # Pass the joint values through the MLP to obtain the final stock price
            x = self.mlp(x)
        else:
            # TODO: x_stock = x[:, :, 0:1]

            # Pass the stock prices throught the LSTM
            x, _ = self.lstm(x, (h_0, c_0))

            # Extract last prediction
            x = x[:, -1, :]

            # Pass stock price through the fully connected layer
            x = self.fc(x)
        
        return x