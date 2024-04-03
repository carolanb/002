import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import ta  


# Función para generar el target basado en el rendimiento futuro del precio
def generate_target(asset_dataframe):
    asset_dataframe['future_price'] = asset_dataframe['Close'].shift(-10)
    asset_dataframe.dropna(inplace=True)
    target_column = []
    for future_price, current_price in zip(asset_dataframe['future_price'], asset_dataframe['Close']):
        if future_price > current_price * 1.005:
            target_column.append(1)
        elif future_price < current_price * 0.995:
            target_column.append(-1)
        else:
            target_column.append(0)

    asset_dataframe['investment_target'] = target_column
    return asset_dataframe

def get_optimal_model_configs(asset_data):
    # Proceso inicial: Generación de objetivo y cálculo de indicadores técnicos
    asset_data = generate_target(asset_data)
    asset_data['percentage_change'] = asset_data['Close'].pct_change()
    asset_data['short_term_sma'] = ta.trend.SMAIndicator(asset_data['Close'], window=5).sma_indicator()
    asset_data['long_term_sma'] = ta.trend.SMAIndicator(asset_data['Close'], window=15).sma_indicator()
    asset_data['relative_strength_index'] = ta.momentum.RSIIndicator(asset_data['Close']).rsi()
    asset_data.drop(['Timestamp', 'Gmtoffset', 'Datetime'], inplace=True, axis=1)
    asset_data.dropna(inplace=True)

    # Ajuste y búsqueda de hiperparámetros para Logistic Regression y SVM
    log_reg_model = LogisticRegression()
    param_grid_log_reg = {'C': [.0001, .001, .01, 0.1, 1, 10, 100]}
    log_reg_grid_search = GridSearchCV(log_reg_model, param_grid_log_reg, scoring='f1_weighted', cv=5)
    log_reg_grid_search.fit(asset_data.drop(['investment_target', 'future_price'], axis=1), asset_data['investment_target'])

    svm_model = SVC()
    svm_grid_search = GridSearchCV(svm_model, param_grid_log_reg, scoring='f1_weighted', cv=5)
    svm_grid_search.fit(asset_data.drop(['investment_target', 'future_price'], axis=1), asset_data['investment_target'])

    # Definición y primera búsqueda de hiperparámetros para GradientBoostingClassifier
    xgb_model = GradientBoostingClassifier()
    param_grid_xgb = {
        'n_estimators': [1, 7, 10, 30, 50, 100, 200, 300, 500, 1000],
        'subsample': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    }
    xgb_grid_search = GridSearchCV(xgb_model, param_grid_xgb, scoring='f1_weighted')
    xgb_grid_search.fit(asset_data.drop(['investment_target', 'future_price'], axis=1), asset_data['investment_target'])

    # Segunda configuración de parámetros para GradientBoostingClassifier
    param_grid_xgb_2 = {
        'learning_rate': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
        'loss': ['log_loss', 'exponential']
    }
    xgb_grid_search_2 = GridSearchCV(xgb_model, param_grid_xgb_2, scoring='f1_weighted')
    xgb_grid_search_2.fit(asset_data.drop(['investment_target', 'future_price'], axis=1), asset_data['investment_target'])

    return {
        'optimal_C_log_reg': log_reg_grid_search.best_params_['C'],
        'optimal_C_svm': svm_grid_search.best_params_['C'],
        'optimal_params_xgb': xgb_grid_search.best_params_,  # Ahora se devuelve directamente
        'optimal_params_xgb_2': xgb_grid_search_2.best_params_,  # Agregado para completitud
    }