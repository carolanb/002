import pandas as pd
import itertools 
import talib as ta
from utils import strategies_design
from utils import execute_buy_order
from utils import update_cash_and_position
from utils import close_position
from utils import update_portfolio_values
from utils import execute_sell_order

def perform(data_o, data_t):
    global  cash, order_count, cash_values, portfolio_values
    cash = 1_000_000
    order_count = 0

    data = data_o.head(1000)  # Asume que esta es una forma de preprocesar tus datos.
    data_validation = data_t.copy()

    
    df_results = pd.DataFrame({'gain': [], 'strategy': []})

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
        return final_portfolio_value  
    
    original_strategies = ['rsi', 'bb', 'MM']

    # Generar todas las combinaciones posibles de estrategias para diferentes longitudes
    all_combinations = []
    for r in range(1, len(original_strategies) + 1):
        combinations = list(itertools.combinations(original_strategies, r))
        all_combinations.extend(combinations)

    # Si necesitas convertir las tuplas a listas (por ejemplo, para que coincidan con el formato de tus estrategias existentes)
    all_combinations = [list(comb) for comb in all_combinations]

    for strat in all_combinations:
        portfolio = backtest(strat)
        df_trash = pd.DataFrame({'gain': [portfolio], 'strategy': [str(strat)]})
        df_results = pd.concat([df_results, df_trash], ignore_index=True)