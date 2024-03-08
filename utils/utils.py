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