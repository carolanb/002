import pandas as pd
from utils import strategies_design
from utils import execute_buy_order
from utils import update_cash_and_position
from utils import close_position


def perform():
    data = pd.read_csv('./data/aapl_5m_train.csv')
    data_val = pd.read_csv('./data/aapl_5m_validation.csv')
    df_results = pd.DataFrame({'gain': [], 'ind': []})

    data =   (data)

    portfolio_values = []
    cash_values = []

    strategies = ['RSI', 'BB', 'SMA']

    def backtest(strat):
        cash = 1_000_000
        df_sell = pd.DataFrame()
        df_buy = pd.DataFrame()
        COMISSION = 0.0025
        STOP_LOSS = 0.04
        TAKE_PROFIT = 0.06
        positions = []
        closed_positions = []

        strategies_design(strat, data, data_validation, df_buy, df_sell)

        # Main loop
        # Main loop
        for (idx, row), (_, row_buy), (_, row_sell) in zip(data_validation.iterrows(), df_buy.iterrows(), df_sell.iterrows()):
            # Update price
            price = 3333 * row.Close

            # Close and open LONG positions
            for position in list(positions):
                if position.is_active:
                    close_position(price, position, positions, closed_positions, COMISSION)
            if row_buy.sum() == len(df_buy.columns):
                execute_buy_order(row, positions, COMISSION, 3333)

            # Close and open SHORT positions
            for position in list(positions):
                if position.is_active:
                    close_position(price, position, positions, closed_positions, COMISSION)
            if row_sell.sum() == len(df_sell.columns):
                execute_sell_order(row, positions, COMISSION, 3333)

            # Update cash and portfolio values for open positions
            update_portfolio_values(data_validation, positions, 3333, 1000)

        # Final cash and portfolio calculation outside the loop if necessary
        final_portfolio_value = update_portfolio_values(data_validation, positions, 3333, 1000)