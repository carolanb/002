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
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers import Input

class Order:
    def __init__(self, timestamp, bought_at, stop_loss, take_profit, order_type, sold_at=None, is_active=True):
        self.timestamp = timestamp
        self.bought_at = bought_at
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.order_type = order_type
        self.sold_at = sold_at
        self.is_active = is_active
 

# TA----------------------------------------------------------------------------------------------- 
def strategies_design(strate, train_data, df_buy, df_sell, rsi_thresholds, bb_window, mm_windows):
    if 'rsi' in strate:
        # Utilizar rsi_thresholds[0] y rsi_thresholds[1] como límites para la compra y venta respectivamente
        train_rsi = RSIIndicator(close=train_data['Close'], window=14).rsi()
        df_buy['rsi_buy_trade_signal'] = train_rsi < rsi_thresholds[0]  # -----
        df_sell['rsi_sell_trade_signal'] = train_rsi > rsi_thresholds[1]  # -----
   
    if 'bb' in strate:
        # Usar bb_window para la ventana de las Bandas de Bollinger
        rolling_mean = train_data['Close'].rolling(window=bb_window).mean()  # -----
        rolling_std = train_data['Close'].rolling(window=bb_window).std()  # -----
        train_data['BBANDS_UpperBand'] = rolling_mean + (rolling_std * 2)
        train_data['BBANDS_LowerBand'] = rolling_mean - (rolling_std * 2)
        df_buy['bb_buy_trade_signal'] = train_data['Close'] < train_data['BBANDS_LowerBand']
        df_sell['bb_sell_trade_signal'] = train_data['Close'] > train_data['BBANDS_UpperBand']
           
    if 'MM' in strate:
        # Usar mm_windows[0] y mm_windows[1] para las ventanas de las Medias Móviles corta y larga
        short_window, long_window = mm_windows  # -----
        short_ma = train_data['Close'].rolling(window=short_window, min_periods=10).mean()
        long_ma = train_data['Close'].rolling(window=long_window, min_periods=10).mean()
        df_buy['mm_buy_trade_signal'] = short_ma > long_ma
        df_sell['mm_sell_trade_signal'] = short_ma < long_ma


# ML-----------------------------------------------------------------------------------------------

def strategies_design_ml(strate, train_data, df_buy, df_sell, model_params):
    X = train_data.drop(['target', 'price_in_10_days'], axis=1)
    y = train_data['target']

    # Estrategia basada en Logistic Regression
    if 'lr' in strate:
        lr = LogisticRegression(C=model_params['LR_C'])
        lr.fit(X, y)
        predictions = lr.predict(X)
        df_buy['lr_buy_signal'] = (predictions == 1)
        df_sell['lr_sell_signal'] = (predictions == -1)

    # Estrategia basada en Support Vector Classifier
    if 'svc' in strate:
        svc = SVC(C=model_params['SVC_C'], probability=True)
        svc.fit(X, y)
        predictions = svc.predict(X)
        df_buy['svc_buy_signal'] = (predictions == 1)
        df_sell['svc_sell_signal'] = (predictions == -1)

    # Estrategia basada en Gradient Boosting Classifier
    if 'xgboost' in strate:
        # Configura un valor predeterminado para 'loss' si no está presente
        loss_value = model_params['XGBOOST_PARAMS'].get('loss', 'deviance')
        xgb = GradientBoostingClassifier(
            n_estimators=model_params['XGBOOST_PARAMS']['n_estimators'],
            subsample=model_params['XGBOOST_PARAMS']['subsample'],
            learning_rate=model_params['XGBOOST_PARAMS']['learning_rate'],
            loss=loss_value)
        
    
        
        xgb.fit(X, y)
        predictions = xgb.predict(X)
        df_buy['xgb_buy_signal'] = (predictions == 1)
        df_sell['xgb_sell_signal'] = (predictions == -1)


# DL-----------------------------------------------------------------------------------------------
def create_dl_model(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))  # Asegurarse de usar Input(shape=...) correctamente
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Adecuado para 3 clases: 0, 1, y 2
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # Correcto para etiquetas como enteros
                  metrics=['accuracy'])
    return model


def strategies_design_dl(strate, train_data, df_buy, df_sell):
    X = train_data.drop(['target', 'price_in_10_days'], axis=1)
    y = train_data['target']

    # Entrenar modelo de deep learning
    model = create_dl_model(X.shape[1])
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # Obtener predicciones del modelo
    predictions = model.predict(X)
    predictions_classes = np.argmax(predictions, axis=1)

    # Verificar las predicciones
    assert set(predictions_classes).issubset({0, 1, 2}), "Las predicciones contienen valores inesperados"

    # Generar señales de compra y venta
    df_buy['dl_buy_signal'] = (predictions_classes == 2)  # Suponiendo 2 para 'subir'
    df_sell['dl_sell_signal'] = (predictions_classes == 0)  # Suponiendo 0 para 'bajar'

    return {'dl_model': model}



# -----------------------------------------------------------------------------------------
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

def execute_sell_order_ml(row, positions, commission, multiplier, STOP_LOSS, TAKE_PROFIT, cash, order_count):
    price = multiplier * row.Close
    if cash >= price * (1 + commission):
        cash += price * (1 - commission)
        new_order = Order(row.name, price, price * (1 + STOP_LOSS), price * (1 - TAKE_PROFIT), 'SHORT')
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
 
