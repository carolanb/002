import pandas as pd

def perform():
    data = pd.read_csv('./files/aapl_5m_train.csv')
    data_val = pd.read_csv('./files/aapl_5m_validation.csv')
    df_results = pd.DataFrame({'gain': [], 'strategy': []})

    data =   (data)

    portfolio_values = []
    cash_values = []

    strategies = ['RSI', 'BB', '']