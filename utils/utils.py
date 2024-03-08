import ta
import numpy as np

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
    if 'rsi':
        # Proceses
        df_buy['rsi_buy_trade_signal'] = [True if cat == 1 else False for cat in rsi]
        df_sell['rsi_sell_trade_signal'] = [True if cat == -1 else False for cat in rsi]
    
    if 'bb' in strat:
        train_data['bb_upper'], train_data['bb_middle'], train_data['bb_lower'] = ta.volatility.bollinger_hband(train_data['close']), ta.volatility.bollinger_mavg(train_data['close']), ta.volatility.bollinger_lband(train_data['close'])
        
        train_data['bb_buy_signal'] = train_data['close'] < train_data['bb_lower']
        train_data['bb_sell_signal'] = train_data['close'] > train_data['bb_upper']
        validate_data['bb_buy_trade_signal'] = train_data['bb_buy_signal'].shift(1)
        validate_data['bb_sell_trade_signal'] = train_data['bb_sell_signal'].shift(1)

    if 'MM' in strat:
        short_window = 40
        long_window = 100

        data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
        data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

        data['Signal_Long'] = 0.0
        data['Signal_Long'][short_window:] = np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1.0, 0.0)
        data['Positions_Long'] = data['Signal_Long'].diff()

        data['Signal_Short'] = 0.0
        data['Signal_Short'][short_window:] = np.where(data['Short_MA'][short_window:] < data['Long_MA'][short_window:], -1.0, 0.0)
        data['Positions_Short'] = data['Signal_Short'].diff()

def close_position(price, position, positions, closed_positions, commission):
            if position.order_type == 'LONG':
                if price <= position.stop_loss or price >= position.take_profit:
                    update_cash_and_position(price, position, positions, closed_positions, commission, profit=price >= position.take_profit)
            elif position.order_type == 'SHORT':
                if price >= position.stop_loss or price <= position.take_profit:
                    update_cash_and_position(price, position, positions, closed_positions, commission, profit=price <= position.take_profit)

def update_cash_and_position(price, position, positions, closed_positions, commission, profit=True):
            global cash  # Assuming 'cash' is a global variable
            if profit:
                cash += price * (1 - commission if position.order_type == 'LONG' else 1 + commission)
            else:
                cash -= price * (1 + commission if position.order_type == 'LONG' else 1 - commission)
            position.is_active = False
            position.sold_at = price
            closed_positions.append(position)
            positions.remove(position)
    