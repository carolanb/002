import pandas as pd
import itertools 
from utils.utils import strategies_design_ml 
from utils.utils import execute_buy_order
from utils.utils import close_position
from utils.utils import update_portfolio_values
from utils.utils import execute_sell_order

def perform():
    initial_cash = 500_000
    df_results = pd.DataFrame({'gain': [], 'strategy': [], 'orders_executed': []})
    strategy_dfs = {}

    original_strategies = ['rsi', 'bb', 'MM']
    all_combinations = [list(comb) for r in range(1, len(original_strategies) + 1) for comb in itertools.combinations(original_strategies, r)]

    # Preprocesamiento de tus datos
    data = pd.read_csv('./data/aapl_5m_train.csv')
    data_validation = pd.read_csv('./data/aapl_5m_test.csv')

     # Nuevo DataFrame para almacenar los valores combinados del portafolio y el efectivo para cada estrategia
    combined_values_df = pd.DataFrame(index=data.index)

    # Define la función backtest dentro de perform.
    def backtest(strat, data, data_validation, initial_cash, initial_order_count):
        df_sell = pd.DataFrame()
        df_buy = pd.DataFrame()
        COMISSION = 0.0025
        STOP_LOSS = 0.05
        TAKE_PROFIT = 0.05

        nah = 0

        cash, order_count = initial_cash, initial_order_count
        positions, closed_positions = [], []
        portfolio_values, cash_values = [], []
 
        # Suponiendo que strategies_design modifica los df_buy y df_sell
        strategies_design_ml(strat, data, data_validation, df_buy, df_sell)

        # DataFrame para registrar valores durante el backtesting
        record_df = pd.DataFrame(index=data.index, columns=['Portfolio Value', 'Cash'])
        
 
        # Bucle principal de backtest
        for (idx, row), (_, row_buy), (_, row_sell) in zip(data.iterrows(), df_buy.iterrows(), df_sell.iterrows()):
            price = 1 * row.Close

            # Cierre y apertura de posiciones LONG y SHORT
            for position in positions[:]:  # Copia de la lista para iteración segura
                if position.is_active:
                    cash = close_position(price, position, COMISSION, cash)

            # Condiciones para ejecutar órdenes de compra/venta
            if row_buy.sum() == len(df_buy.columns):
                cash, order_count = execute_buy_order(row, positions, COMISSION, 1, STOP_LOSS, TAKE_PROFIT, cash, order_count)

            if row_sell.sum() == len(df_sell.columns):
                cash, order_count = execute_sell_order(row, positions, COMISSION, 1, STOP_LOSS, TAKE_PROFIT, cash, order_count)
               
            current_portfolio_value = update_portfolio_values(data, positions, portfolio_values,nah, 1)
            portfolio_values.append(current_portfolio_value)
            cash_values.append(cash)

            active_positions_value = sum(order.bought_at for order in positions if order.is_active)
            record_df.at[idx, 'Portfolio Value'] = active_positions_value + cash  # Suma del valor de posiciones activas más el efectivo
            record_df.at[idx, 'Cash'] = cash
            
        return cash, order_count, record_df, df_buy, df_sell
    
    # Bucle a través de todas las combinaciones de estrategias
    for strat in all_combinations:
        cash, order_count, record_df, df_buy, df_sell = backtest(
            strat, data, data_validation, initial_cash, 0)
        
        # Almacenar resultados y actualizar DataFrames
        final_value = record_df['Portfolio Value'].iloc[-1] + record_df['Cash'].iloc[-1]  # Suma de valor de portafolio final y efectivo
        # Crear un DataFrame temporal con los nuevos datos
        new_row = pd.DataFrame({'gain': [final_value], 'strategy': [str(strat)], 'orders_executed': [order_count]})
        # Concatenar el nuevo DataFrame con df_results
        df_results = pd.concat([df_results, new_row], ignore_index=True)
        strategy_dfs[str(strat)] = {'df_buy': df_buy, 'df_sell': df_sell, 'records': record_df}
        combined_values_df[str(strat)] = record_df['Portfolio Value'] + record_df['Cash']  # Suma de valores de portafolio y efectivo en cada paso

    return df_results, strategy_dfs, combined_values_df