import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import ta  


def y_generator(data):
    data['price_in_10_days'] = data['Close'].shift(-10)
    data.dropna(inplace=True)
    y_target = []
    for price_10, clos in zip(data['price_in_10_days'], data['Close']):
        if price_10 > clos * 1.005:
            y_target.append(2)  # Cambiado a 2 para 'subir'
        elif price_10 < clos * 0.995:
            y_target.append(0)  # Mantenido como 0 para 'bajar'
        else:
            y_target.append(1)  # Cambiado a 1 para 'mantener'
    data['target'] = y_target
    return data

def prepare_and_optimize(data):
    # Preparación de datos
    data = y_generator(data)
    data['rend'] = data['Close'].pct_change()
    data['short_sma'] = ta.trend.SMAIndicator(data['Close'], window=5).sma_indicator()
    data['long_sma'] = ta.trend.SMAIndicator(data['Close'], window=15).sma_indicator()
    data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    
    # Asegúrate de excluir cualquier columna de fecha/hora o no numérica aquí
    data = data.select_dtypes(include=['number'])
    data.dropna(inplace=True)

    X = data.drop(['target', 'price_in_10_days'], axis=1)
    y = data['target']

    # Optimización para Logistic Regression
    lr = LogisticRegression(max_iter=10000)
    param_grid_lr = {'C': [.0001, .001, .01, 0.1, 1, 10, 100]}
    grid_search_lr = GridSearchCV(lr, param_grid_lr, scoring='f1_weighted', cv=5)
    grid_search_lr.fit(X, y)
    best_C_lr = grid_search_lr.best_params_['C']

    # Optimización para SVC
    svc = SVC(max_iter=10000)
    grid_search_svc = GridSearchCV(svc, param_grid_lr, scoring='f1_weighted', cv=5)
    grid_search_svc.fit(X, y)
    best_C_svc = grid_search_svc.best_params_['C']

    # Optimización para Gradient Boosting Classifier
    xgb = GradientBoostingClassifier()
    param_grid_gb = {
        'n_estimators': [10, 50, 100, 200],
        'subsample': [0.5, 0.7, 1.0],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'loss': ['log_loss', 'exponential']  # Cambiado de 'deviance' a 'log_loss'
    }

    grid_search_gb = GridSearchCV(xgb, param_grid_gb, scoring='f1_weighted', cv=5)
    grid_search_gb.fit(X, y)
    best_params_gb = grid_search_gb.best_params_

    # Retorna el DataFrame preparado y los mejores parámetros encontrados
    return data, {
        'LR_C': best_C_lr, 
        'SVC_C': best_C_svc, 
        'XGBOOST_PARAMS': best_params_gb
    }

def prepare(data):
    # Preparación de datos
    data = y_generator(data)
    data['rend'] = data['Close'].pct_change()
    data['short_sma'] = ta.trend.SMAIndicator(data['Close'], window=5).sma_indicator()
    data['long_sma'] = ta.trend.SMAIndicator(data['Close'], window=15).sma_indicator()
    data['rsi'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    
    # Asegúrate de excluir cualquier columna de fecha/hora o no numérica aquí
    data = data.select_dtypes(include=['number'])
    data.dropna(inplace=True)
    # Retorna el DataFrame preparado 
    return data