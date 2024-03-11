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
    
    portfolio_values = [] 
    cash_values = [] 

    # Define la función backtest dentro de perform.
    def backtest(strat):
        df_sell = pd.DataFrame()
        df_buy = pd.DataFrame()
        COMISSION = 0.0025
        STOP_LOSS = 0.04
        TAKE_PROFIT = 0.06
        positions = []
        closed_positions = []

        # Suponiendo que strategies_design modifica los df_buy y df_sell
        strategies_design(strat, data, data_validation, df_buy, df_sell)

        # Bucle principal de backtest
        for (idx, row), (_, row_buy), (_, row_sell) in zip(data.iterrows(), df_buy.iterrows(), df_sell.iterrows()):
            price = 1 * row.Close  # Precio actual según la simulación

            # Cierre y apertura de posiciones LONG y SHORT
            for position in list(positions):
                if position.is_active:
                    close_position(price, position, positions, closed_positions, COMISSION)

            if row_buy.sum() == len(df_buy.columns):
                execute_buy_order(row, positions, COMISSION, 1, STOP_LOSS, TAKE_PROFIT)
               
            if row_sell.sum() == len(df_sell.columns):
                execute_sell_order(row, positions, COMISSION, 1, STOP_LOSS, TAKE_PROFIT)
               
            # Actualiza los valores de efectivo y del portafolio para las posiciones abiertas
            update_portfolio_values(data, positions, portfolio_values, cash, 1, 1000)

        # Cálculo final del valor del portafolio fuera del bucle principal
        final_portfolio_value = update_portfolio_values(data, positions, portfolio_values, cash, 1, 1000)
        return final_portfolio_value, df_buy, df_sell, order_count  

    original_strategies = ['rsi', 'bb', 'MM']

    # Generar todas las combinaciones posibles de estrategias
    all_combinations = [list(comb) for r in range(1, len(original_strategies) + 1) for comb in itertools.combinations(original_strategies, r)]

    strategy_dfs = {}  # Diccionario para almacenar los resultados de cada estrategia.

    for strat in all_combinations:
        portfolio, strategy_df_buy, strategy_df_sell, orders_executed = backtest(strat)  # Asegúrate de capturar orders_executed
        df_trash = pd.DataFrame({'gain': [portfolio], 'strategy': [str(strat)], 'orders_executed': [orders_executed]})
        df_results = pd.concat([df_results, df_trash], ignore_index=True)

        # Almacena los dataframes de compra y venta actualizados para cada estrategia.
        strategy_dfs[str(strat)] = {'df_buy': strategy_df_buy.copy(), 'df_sell': strategy_df_sell.copy()}

    # Devolver el DataFrame de resultados podría ser útil para el análisis posterior.
    return df_results, strategy_dfs