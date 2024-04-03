import pandas as pd
import itertools 
from utils.utils import define_strategies_ml 
from utils.utils import execute_buy_order
from utils.utils import close_position
from utils.utils import update_portfolio_values
from utils.utils import execute_sell_order

# main
def perform(data, commission, stop_loss, take_profit):
    initial_cash = 500_000
    df_results = pd.DataFrame({'gain': [], 'strategy': [], 'orders_executed': []})
    strategy_dfs = {}

    original_strategies = ['svc', 'lr', 'xgboost']
    all_combinations = [list(comb) for r in range(1, len(original_strategies) + 1) for comb in itertools.combinations(original_strategies, r)]

    combined_values_df = pd.DataFrame(index=data.index)

    def backtest(strat, data, initial_cash, commission, stop_loss, take_profit):
        df_buy = pd.DataFrame(index=data.index)
        df_sell = pd.DataFrame(index=data.index)
        cash = initial_cash
        order_count = 0
        positions = []
        portfolio_values = []

        define_strategies_ml(strat, data[:-len(data)//5], data[-len(data)//5:], df_buy, df_sell)

        for (idx, row), (_, row_buy), (_, row_sell) in zip(data.iterrows(), df_buy.iterrows(), df_sell.iterrows()):
            price = row['Close']
            for position in positions[:]:
                if position.is_active:
                    cash = close_position(price, position, commission, cash)

            if row_buy['xgb_buy_signal'] or row_buy['svc_buy_signal'] or row_buy['lr_buy_signal']:
                cash, order_count = execute_buy_order(row, positions, commission, stop_loss, take_profit, cash, order_count)

            if row_sell['xgb_sell_signal'] or row_sell['svc_sell_signal'] or row_sell['lr_sell_signal']:
                cash, order_count = execute_sell_order(row, positions, commission, stop_loss, take_profit, cash, order_count)

            portfolio_values.append(cash)

        return cash, order_count, portfolio_values, df_buy, df_sell

    for strat in all_combinations:
        final_cash, order_count, portfolio_values, df_buy, df_sell = backtest(
            strat, data, initial_cash, commission, stop_loss, take_profit)

        final_value = portfolio_values[-1]
        new_row = pd.DataFrame({'gain': [final_value], 'strategy': [str(strat)], 'orders_executed': [order_count]})
        df_results = pd.concat([df_results, new_row], ignore_index=True)
        strategy_dfs[str(strat)] = {'df_buy': df_buy, 'df_sell': df_sell, 'portfolio_values': portfolio_values}
        combined_values_df[str(strat)] = portfolio_values

    return df_results, strategy_dfs, combined_values_df
