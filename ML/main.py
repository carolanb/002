import pandas as pd
import itertools 
from utils.utils import strategies_design_ml 
from utils.ml import prepare_and_optimize 
from utils.utils import execute_buy_order
from utils.utils import close_position
from utils.utils import update_portfolio_values
from utils.utils import execute_sell_order


# main
def perform(data, commission, stop_loss, take_profit):
    initial_cash = 500_000
    initial_short_cash = 500_000
    df_results = pd.DataFrame({'gain': [], 'strategy': [], 'orders_executed': []})
    strategy_dfs = {}

    # Preparación y optimización de los datos y modelos
    data, model_params = prepare_and_optimize(data)

    # original_strategies = ['lr']
    # original_strategies = ['svc']
    # original_strategies = ['xgboost']
    # original_strategies = ['lr' , 'svc']
    original_strategies = ['lr' , 'svc', 'xgboost'] 
    
    all_combinations = [list(comb) for r in range(1, len(original_strategies) + 1) for comb in itertools.combinations(original_strategies, r)]

    combined_values_df = pd.DataFrame(index=data.index)

    # Asegúrate de que las funciones necesarias para ejecutar órdenes y actualizar valores estén definidas

    def backtest(strat, data, initial_cash, initial_short_cash, initial_order_count, commission, stop_loss, take_profit, model_params):
        df_sell = pd.DataFrame(index=data.index)
        df_buy = pd.DataFrame(index=data.index)
        cash, short_cash, order_count = initial_cash, initial_short_cash, initial_order_count
        positions = []
        portfolio_values = []
        COMISSION = 0.0025
        STOP_LOSS = 0.05
        TAKE_PROFIT = 0.05        

        # Asegúrate de que strategies_design esté adaptada para utilizar los model_params
        strategies_design_ml(strat, data, df_buy, df_sell, model_params)

        # DataFrame para registrar valores durante el backtesting
        record_df = pd.DataFrame(index=data.index, columns=['Portfolio Value', 'Cash'])

        # Bucle principal de backtest
        for (idx, row), (_, row_buy), (_, row_sell) in zip(data.iterrows(), df_buy.iterrows(), df_sell.iterrows()):
            price = 1 * row.Close

            # Cierre y apertura de posiciones LONG y SHORT
            for position in positions[:]:  # Copia de la lista para iteración segura
                if position.is_active:
                    if position.order_type == 'LONG':
                        cash = close_position(price, position, COMISSION, cash)
                        position.is_active = False  # Asegúrate de marcar la posición como no activa

                    elif position.order_type == 'SHORT':
                        short_cash = close_position(price, position, COMISSION, short_cash)
                        position.is_active = False  # Asegúrate de marcar la posición como no activa

            # Contar posiciones activas
            active_positions = sum(1 for position in positions if position.is_active)


            # Condiciones para ejecutar órdenes de compra/venta
            if active_positions < 100 and row_buy.sum() == len(df_buy.columns):
                cash, order_count = execute_buy_order(row, positions, COMISSION, 1, STOP_LOSS, TAKE_PROFIT, cash, order_count)

            if active_positions < 100 and row_sell.sum() == len(df_sell.columns):
                short_cash, order_count = execute_sell_order(row, positions, COMISSION, 1, STOP_LOSS, TAKE_PROFIT, short_cash, order_count)

            current_portfolio_value = update_portfolio_values(data, positions, portfolio_values, cash + short_cash, 1)
            portfolio_values.append(current_portfolio_value)

            record_df.at[idx, 'Portfolio Value'] = current_portfolio_value
            record_df.at[idx, 'Cash'] = cash + short_cash
            
        return cash + short_cash, order_count, record_df, df_buy, df_sell

    # Bucle a través de todas las combinaciones de estrategias
    for strat in all_combinations:
        final_cash, order_count, record_df, df_buy, df_sell = backtest(
            strat, data, initial_cash, initial_short_cash, 0,
            commission, stop_loss, take_profit, model_params
        )

        final_value = record_df['Portfolio Value'].iloc[-1]
        new_row = pd.DataFrame({'gain': [final_value], 'strategy': [str(strat)], 'orders_executed': [order_count]})
        df_results = pd.concat([df_results, new_row], ignore_index=True)
        strategy_dfs[str(strat)] = {'df_buy': df_buy, 'df_sell': df_sell, 'records': record_df}
        combined_values_df[str(strat)] = record_df['Portfolio Value']

    return df_results, strategy_dfs, combined_values_df
