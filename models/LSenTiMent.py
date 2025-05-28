import torch
from torch import nn

class LSenTiMent(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, window_size, mlp_layers, mlp_hidden_size, dropout_rate, sentiment = True):
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
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, 1)
        )

        self.sentiment_cnn = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                # nn.Conv1d(32, 32, kernel_size=3, padding=1),
                # nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)  # compress time dimension
            )
        self.sentiment_fc = nn.Linear(32, hidden_size)

        # Final multi-layer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        for _ in range(mlp_layers):
            self.mlp.append(nn.Linear(mlp_hidden_size, mlp_hidden_size))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(p=dropout_rate))

        self.mlp.append(nn.Linear(mlp_hidden_size, 1))

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
            
            # Extract last prediction
            x_stock = x_stock[:, -1, :]

            # Pass sentiment through a CNN
            x_sentiment = x_sentiment.permute(0, 2, 1)
            x_sentiment = self.sentiment_cnn(x_sentiment).squeeze(-1)
            x_sentiment = self.sentiment_fc(x_sentiment)

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