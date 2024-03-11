from ta.momentum import RSIIndicator
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import ta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ta.momentum import RSIIndicator
import itertools
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from .ml import optimal_C_log_reg, optimal_C_svm, optimal_params_xgb


class Order:
    def __init__(self, timestamp, bought_at, stop_loss, take_profit, order_type, sold_at=None, is_active=True):
        self.timestamp = timestamp
        self.bought_at = bought_at
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.order_type = order_type
        self.sold_at = sold_at
        self.is_active = is_active
 
def strategies_design(strate, train_data, validate_data, df_buy, df_sell):
    if 'rsi' in strate:
        train_rsi = RSIIndicator(close=train_data['Close'], window=14).rsi()
        validate_rsi = RSIIndicator(close=validate_data['Close'], window=14).rsi()
        df_buy['rsi_buy_trade_signal'] = train_rsi < 25  
        df_sell['rsi_sell_trade_signal'] = train_rsi > 75  
   
    if 'bb' in strate:
            # Calcular medias móviles y desviación estándar para los datos de validación
            rolling_mean = train_data['Close'].rolling(window=20).mean()
            rolling_std = train_data['Close'].rolling(window=20).std()
 
            # Calcular las Bandas de Bollinger para los datos de validación
            train_data['BBANDS_UpperBand'] = rolling_mean + (rolling_std * 2)
            train_data['BBANDS_LowerBand'] = rolling_mean - (rolling_std * 2)
 
            # Generar señales de compra y venta basadas en las Bandas de Bollinger para la validación
            df_buy['bb_buy_trade_signal'] = train_data['Close'] < train_data['BBANDS_LowerBand']
            df_sell['bb_sell_trade_signal'] = train_data['Close'] > train_data['BBANDS_UpperBand']
           
    if 'MM' in strate:
        short_window, long_window = 40, 100
        short_ma = train_data['Close'].rolling(window=short_window, min_periods=10).mean()
        long_ma = train_data['Close'].rolling(window=long_window, min_periods=10).mean()
        df_buy['mm_buy_trade_signal'] = short_ma > long_ma  # Compra cuando la media corta cruza por encima de la media larga
        df_sell['mm_sell_trade_signal'] = short_ma < long_ma  # Venta cuando la media corta cruza por debajo de la media larga
 

def define_strategies_ml(strategy_list, historical_data, validation_data, buy_signals, sell_signals):
    if 'svc' in strategy_list:
        svc_optimal = SVC(C=optimal_C_svm)
        svc_optimal.fit(historical_data.drop(['investment_target', 'future_price', 'Timestamp', 'Gmtoffset', 'Datetime'], axis=1),
                        historical_data['investment_target'])
        svc_predictions = svc_optimal.predict(
            validation_data.drop(['investment_target', 'future_price', 'Timestamp', 'Gmtoffset', 'Datetime'], axis=1))
        buy_signals['svc_buy_signal'] = [True if prediction == 1 else False for prediction in svc_predictions]
        sell_signals['svc_sell_signal'] = [True if prediction == -1 else False for prediction in svc_predictions]

    if 'lr' in strategy_list:
        lr_optimal = LogisticRegression(C=optimal_C_log_reg)
        lr_optimal.fit(historical_data.drop(['investment_target', 'future_price', 'Timestamp', 'Gmtoffset', 'Datetime'], axis=1),
                       historical_data['investment_target'])
        lr_predictions = lr_optimal.predict(
            validation_data.drop(['investment_target', 'future_price', 'Timestamp', 'Gmtoffset', 'Datetime'], axis=1))
        buy_signals['lr_buy_signal'] = [True if prediction == 1 else False for prediction in lr_predictions]
        sell_signals['lr_sell_signal'] = [True if prediction == -1 else False for prediction in lr_predictions]

    if 'xgboost' in strategy_list:
        xgb_optimal = GradientBoostingClassifier(learning_rate=optimal_params_xgb['learning_rate'], n_estimators=optimal_params_xgb['n_estimators'],
                                                 subsample=optimal_params_xgb['subsample'], loss=optimal_params_xgb['loss'])
        xgb_optimal.fit(historical_data.drop(['investment_target', 'future_price', 'Timestamp', 'Gmtoffset', 'Datetime'], axis=1),
                        historical_data['investment_target'])
        xgb_predictions = xgb_optimal.predict(
            validation_data.drop(['investment_target', 'future_price', 'Timestamp', 'Gmtoffset', 'Datetime'], axis=1))
        buy_signals['xgb_buy_signal'] = [True if prediction == 1 else False for prediction in xgb_predictions]
        sell_signals['xgb_sell_signal'] = [True if prediction == -1 else False for prediction in xgb_predictions]

def update_cash_and_positions(price, position, commission, cash, profit=True):
    if profit:
        cash += (price - position.bought_at) * (1 - commission if position.order_type == 'LONG' else 1 + commission)
    else:
        cash -= (position.bought_at - price) * (1 + commission if position.order_type == 'LONG' else 1 - commission)
    position.is_active = False
    position.sold_at = price
    # No necesitas pasar ni modificar 'positions' ni 'closed_positions' aquí; eso debería manejarse en otro lugar
    return cash  # Devuelve el 'cash' actualizado


def close_position(price, position, commission, cash):
    if position.order_type == 'LONG':
        if price <= position.stop_loss or price >= position.take_profit:
            cash = update_cash_and_positions(price, position, commission, cash, profit=price >= position.take_profit)
    elif position.order_type == 'SHORT':
        if price >= position.stop_loss or price <= position.take_profit:
            cash = update_cash_and_positions(price, position, commission, cash, profit=price <= position.take_profit)
    return cash  # Devuelve el 'cash' actualizado

def execute_buy_order(row, positions, commission, multiplier, STOP_LOSS, TAKE_PROFIT, cash, order_count):
    price = multiplier * row['Close']
    if cash >= price * (1 + commission):
        cash -= price * (1 + commission)
        new_order = Order(row.name, price, price * (1 - STOP_LOSS), price * (1 + TAKE_PROFIT), 'LONG')
        positions.append(new_order)
        order_count += 1
    return cash, order_count

def execute_sell_order(row, positions, commission, multiplier, STOP_LOSS, TAKE_PROFIT, cash, order_count):
    price = multiplier * row.Close
    if cash >= price * (1 + commission):
        cash += price * (1 - commission)
        new_order = Order(row.Timestamp, price, price * (1 + STOP_LOSS), price * (1 - TAKE_PROFIT), 'SHORT')
        positions.append(new_order)
        order_count += 1
    return cash, order_count
 
def update_portfolio_values(data, positions, portfolio_values, cash, multiplier):
    open_long_positions = [multiplier * data.Close.iloc[-1] for position in positions if position.order_type == 'LONG' and position.is_active]
    open_short_positions = [-data.Close.iloc[-1] * multiplier for position in positions if position.order_type == 'SHORT' and position.is_active]
    portfolio_values.append(sum(open_long_positions) + sum(open_short_positions) + cash)
    return sum(open_long_positions) + sum(open_short_positions) + cash  
 
