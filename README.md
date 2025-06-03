# <center>Stock Price Prediction with Social Media Sentiment Data</center>

This project was developed during the **Deep learning** course at the **Data Science** master's programme at the **Faculty of Computer and Information Science, University of Ljubljana**.

More information can be found in the project report in `report/`.

### Abstract

In this project we deal with stock price prediction for MAANG companies. We show how this
is done using an LSTM network and develop a baseline model. As an addition in hopes to improve the
predictions, we incorporate sentiments from Twitter/X posts. We show how we extracted sentiments
using the FinBERT model. Then we develop two different architectures combining the baseline model
with the sentiments. One uses a single LSTM with more input features, and the other has two LSTM-s,
one for stock prices and one for sentiments. We evaluate the results of all three models on the test data
set for each stock separately and discuss how our approach could be improved.

## Setup
0. Install Python. This project was developed with Python 3.12.
1. Clone this repository.
2. Create a new Python virtual environment with `python -m venv .venv`.
3. Activate the environment with `source .venv/bin/activate` on Mac and Linux or with `.venv/Scripts/activate.bat` on Windows.
4. Install the requirements with `pip install -r requirements.txt`.

## Inference
An example of how to run inference can be found in the notebook `notebooks/evaluation.ipynb`.

## Training
You can train the model by running the script `src/train_test.py`. To get a list of parameters use
```sh
python src/train_test.py --help
```

The weights of the trained models can be found in `models/final`.