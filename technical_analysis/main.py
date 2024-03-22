import pandas as pd
import itertools 
from utils.utils import strategies_design 
from utils.utils import execute_buy_order
from utils.utils import update_cash_and_positions
from utils.utils import close_position
from utils.utils import update_portfolio_values
from utils.utils import execute_sell_order

# main
def perform(data, rsi_thresholds, bb_window, mm_windows, commission, stop_loss, take_profit):
    print(2)
    
    # initial_cash = 500_000
    # initial_short_cash = 500_000 
    # df_results = pd.DataFrame({'gain': [], 'strategy': [], 'orders_executed': []})
    # strategy_dfs = {}

    # original_strategies = ['rsi', 'bb', 'MM']
    # all_combinations = [list(comb) for r in range(1, len(original_strategies) + 1) for comb in itertools.combinations(original_strategies, r)]

    # # Nuevo DataFrame para almacenar los valores combinados del portafolio y el efectivo para cada estrategia
    # combined_values_df = pd.DataFrame(index=data.index)

    # # Define la función backtest dentro de perform.
    # def backtest(strat, data, initial_cash, initial_short_cash, initial_order_count, rsi_thresholds, bb_window, mm_windows, commission, stop_loss, take_profit):  # -----
    #     df_sell = pd.DataFrame()
    #     df_buy = pd.DataFrame()
    #     COMISSION = 0.0025
    #     STOP_LOSS = stop_loss  # Usar el parámetro 'stop_loss' -----
    #     TAKE_PROFIT = take_profit  # Usar el parámetro 'take_profit' -----


    #     cash, short_cash, order_count = initial_cash, initial_short_cash, initial_order_count
    #     positions = []
    #     portfolio_values = []

    #     # Suponiendo que strategies_design modifica los df_buy y df_sell
    #     strategies_design(strat, data, df_buy, df_sell, rsi_thresholds, bb_window, mm_windows)  # Asegúrate de que esto está correcto -----
    #     strategies_design(strat, data, df_buy, df_sell, rsi_thresholds, bb_window, mm_windows)  # Asegúrate de pasar todos los argumentos

    #     # DataFrame para registrar valores durante el backtesting
    #     record_df = pd.DataFrame(index=data.index, columns=['Portfolio Value', 'Cash'])

    #     # Bucle principal de backtest
    #     for (idx, row), (_, row_buy), (_, row_sell) in zip(data.iterrows(), df_buy.iterrows(), df_sell.iterrows()):
    #         price = 1 * row.Close

    #         # Cierre y apertura de posiciones LONG y SHORT
    #         for position in positions[:]:  # Copia de la lista para iteración segura
    #             if position.is_active:
    #                 if position.order_type == 'LONG':
    #                     cash = close_position(price, position, COMISSION, cash)
    #                 elif position.order_type == 'SHORT':
    #                     short_cash = close_position(price, position, COMISSION, short_cash)

    #         # Condiciones para ejecutar órdenes de compra/venta
    #         if row_buy.sum() == len(df_buy.columns):
    #             cash, order_count = execute_buy_order(row, positions, COMISSION, 1, STOP_LOSS, TAKE_PROFIT, cash, order_count)

    #         if row_sell.sum() == len(df_sell.columns):
    #             short_cash, order_count = execute_sell_order(row, positions, COMISSION, 1, STOP_LOSS, TAKE_PROFIT, short_cash, order_count)

    #         current_portfolio_value = update_portfolio_values(data, positions, portfolio_values, cash + short_cash, 1)
    #         portfolio_values.append(current_portfolio_value)

    #         record_df.at[idx, 'Portfolio Value'] = current_portfolio_value
    #         record_df.at[idx, 'Cash'] = cash + short_cash
            
    #     return cash + short_cash, order_count, record_df, df_buy, df_sell

    # # Bucle a través de todas las combinaciones de estrategias
    # for strat in all_combinations:
    #     final_cash, order_count, record_df, df_buy, df_sell = backtest(
    #         strat, data, initial_cash, initial_short_cash, 0, 
    #         rsi_thresholds, bb_window, mm_windows, commission, stop_loss, take_profit)  # -----

    #     final_value = record_df['Portfolio Value'].iloc[-1]
    #     new_row = pd.DataFrame({'gain': [final_value], 'strategy': [str(strat)], 'orders_executed': [order_count]})
    #     df_results = pd.concat([df_results, new_row], ignore_index=True)
    #     strategy_dfs[str(strat)] = {'df_buy': df_buy, 'df_sell': df_sell, 'records': record_df}
    #     combined_values_df[str(strat)] = record_df['Portfolio Value']  # Solo usa 'Portfolio Value' ya que representa la suma total de valores

    # return df_results, strategy_dfs, combined_values_df


def optimization_function(train, data):
    # Definir el rango de los hiperparámetros para que Optuna elija
    rsi_buy = train.suggest_int('rsi_buy', 10, 40)
    rsi_sell = train.suggest_int('rsi_sell', 60, 90)
    bb_window = train.suggest_int('bb_window', 15, 25)
    mm_short_window = train.suggest_int('mm_short_window', 20, 50)
    mm_long_window = train.suggest_int('mm_long_window', 100, 200)
    commission = train.suggest_float('commission', 0.001, 0.005)
    stop_loss = train.suggest_float('stop_loss', 0.02, 0.1)
    take_profit = train.suggest_float('take_profit', 0.02, 0.1)

    # Utiliza el conjunto de datos de entrenamiento para realizar el backtest y encontrar el mejor conjunto de hiperparámetros
    df_results, _, _ = perform(
        data,  # Solo utiliza 'data' aquí, que es tu conjunto de entrenamiento
        rsi_thresholds=(rsi_buy, rsi_sell),
        bb_window=bb_window,
        mm_windows=(mm_short_window, mm_long_window),
        commission=commission,
        stop_loss=stop_loss,
        take_profit=take_profit
    )
    # Maximizar el valor final del portafolio
    final_value = df_results['gain'].max()
    return -final_value