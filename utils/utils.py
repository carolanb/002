import ta

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