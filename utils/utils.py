import ta  
import numpy as np
from ta.momentum import RSIIndicator

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
        df_buy['rsi_buy_trade_signal'] = train_rsi < 30  
        df_sell['rsi_sell_trade_signal'] = train_rsi > 70  
    
    if 'bb' in strate:
        rolling_mean = train_data['Close'].rolling(window=20).mean()
        rolling_std = train_data['Close'].rolling(window=20).std()

        train_data['BBANDS_UpperBand'] = rolling_mean + (rolling_std * 2)
        train_data['BBANDS_LowerBand'] = rolling_mean - (rolling_std * 2)

        df_buy['bb_buy_trade_signal'] = train_data['Close'] < train_data['BBANDS_LowerBand']
        df_sell['bb_sell_trade_signal'] = train_data['Close'] > train_data['BBANDS_UpperBand']

    if 'MM' in strate:
        short_window, long_window = 40, 100
        short_ma = train_data['Close'].rolling(window=short_window, min_periods=10).mean()
        long_ma = train_data['Close'].rolling(window=long_window, min_periods=10).mean()
        df_buy['mm_buy_trade_signal'] = short_ma > long_ma  # Compra cuando la media corta cruza por encima de la media larga
        df_sell['mm_sell_trade_signal'] = short_ma < long_ma  # Venta cuando la media corta cruza por debajo de la media larga

def close_position(price, position, positions, closed_positions, commission):
    if position.order_type == 'LONG':
        if price <= position.stop_loss or price >= position.take_profit:
            update_cash_and_position(price, position, positions, closed_positions, commission, profit=price >= position.take_profit)
    elif position.order_type == 'SHORT':
        if price >= position.stop_loss or price <= position.take_profit:
            update_cash_and_position(price, position, positions, closed_positions, commission, profit=price <= position.take_profit)

def update_cash_and_positions(price, position, positions, closed_positions, commission, profit=True):
            global cash  # Assuming 'cash' is a global variable
            if profit:
                cash += price * (1 - commission if position.order_type == 'LONG' else 1 + commission)
            else:
                 cash -= (position.bought_at - price) * (1 + commission if position.order_type == 'LONG' else 1 - commission)
            position.is_active = False
            position.sold_at = price
            closed_positions.append(position)
            positions.remove(position)
            
def execute_buy_order(row, positions, commission, multiplier, STOP_LOSS, TAKE_PROFIT):
            global cash, order count  # Usa variables globales
            price = multiplier * row['Close'] #usa close directamente del row
            if cash >= price * (1 + commission):  # Verifica si tienes suficiente efectivo
                cash -= price * (1 + commission)  # Actualiza el efectivo
                # Crea una nueva orden y la añade a la lista de posiciones
                new_order = Order(row.name, price, price * (1 - STOP_LOSS), price * (1 + TAKE_PROFIT), 'LONG')
                positions.append(new_order)
                order_count += 1  # Incrementa el contador de órdenes 
    
def update_portfolio_values(data_validation, positions, multiplier, short_multiplier):
    open_long_positions = [multiplier * data_validation.Close.iloc[-1] for position in positions if position.order_type == 'LONG' and position.is_active]
    open_short_positions = [-data_validation.Close.iloc[-1] * short_multiplier for position in positions if position.order_type == 'SHORT' and position.is_active]

    # Update global cash and portfolio values
    global cash_values, portfolio_values  
    portfolio_values.append(sum(open_long_positions) + sum(open_short_positions) + cash)
    return sum(open_long_positions) + sum(open_short_positions) + cash  

def execute_sell_order(row, positions, commission, multiplier):
    global cash  # Assuming 'cash' is a global variable
    price = multiplier * row.Close
    if cash >= price * (1 + commission):
        cash += price * (1 - commission)  # This seems like a mistake, it should reduce cash, but as it's a SHORT sell, it's adding cash temporarily.
        order = Order(timestamp=row.Timestamp,
                      bought_at=price,
                      stop_loss=price * (1 + STOP_LOSS),
                      take_profit=price * (1 - TAKE_PROFIT),
                      order_type='SHORT')
        positions.append(order)

