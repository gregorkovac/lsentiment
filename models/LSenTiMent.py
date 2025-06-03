import torch
from torch import nn

class LSenTiMent(nn.Module):
    def __init__(self, input_size, hidden_size1, num_layers1, hidden_size2, num_layers2, window_size, mlp_layers, mlp_hidden_size, dropout_rate, sentiment = True, double = False):
        super().__init__()

        # Save parameters
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.num_layers1 = num_layers1
        self.hidden_size2 = hidden_size2
        self.num_layers2 = num_layers2
        self.window_size = window_size
        self.sentiment = sentiment
        self.double_ = double

        if sentiment and not double:
            lstm_stock_input_size = 4
            sentiment = False
        else:
            lstm_stock_input_size = 1

        # LSTM for stock price
        self.lstm_stock = nn.LSTM(input_size=lstm_stock_input_size,
                            hidden_size=hidden_size1,
                            num_layers=num_layers1,
                            batch_first=True)
        
        # Fully-connected layer for stock price LSTM
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size1, 1)
        )

        self.lstm_sentiment = nn.LSTM(input_size=3,
                            hidden_size=hidden_size2,
                            num_layers=num_layers2,
                            batch_first=True)
        
        # Final multi-layer perceptron
        if mlp_layers == 0:
            self.mlp = nn.Linear(hidden_size1 + hidden_size2, 1)
        else:
            self.mlp = [
                nn.Linear(hidden_size1 + hidden_size2, mlp_hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ]

            for _ in range(mlp_layers):
                self.mlp.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
                self.mlp.append(nn.ReLU())
                self.mlp.append(nn.Dropout(p=dropout_rate))

            self.mlp.append(nn.Linear(mlp_hidden_size, 1))
            self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        # Extract batch size
        batch_size = x.size(0)

        # Initialize hidden and cell states for LSTM
        h_0 = torch.zeros(self.num_layers1, batch_size, self.hidden_size1).to(x.device)
        c_0 = torch.zeros(self.num_layers1, batch_size, self.hidden_size1).to(x.device)

        if self.sentiment and self.double_:
            # Extract stock prices in time window
            x_stock = x[:, :, 0:1]

            # Extract sentiments in time window
            x_sentiment = x[:, :, 1:4]

            # Obtain stock results from LSTM
            x_stock, (h_stock, _) = self.lstm_stock(x_stock, (h_0, c_0))

            h_0_sent = torch.zeros(self.num_layers2, batch_size, self.hidden_size2).to(x.device)
            c_0_sent = torch.zeros(self.num_layers2, batch_size, self.hidden_size2).to(x.device)

            x_sentiment, (h_sentiment, _) = self.lstm_sentiment(x_sentiment, (h_0_sent, c_0_sent))

            x_stock = h_stock[-1]
            x_sentiment = h_sentiment[-1]

            # Concat stock and sentiment features
            x = torch.cat((x_stock, x_sentiment), dim=1)

            # Pass the joint values through the MLP to obtain the final stock price
            x = self.mlp(x)
        else:
            # Pass the stock prices throught the LSTM
            x, _ = self.lstm_stock(x, (h_0, c_0))

            # Extract last prediction
            x = x[:, -1, :]

            # Pass stock price through the fully connected layer
            x = self.fc(x)
        
        return x