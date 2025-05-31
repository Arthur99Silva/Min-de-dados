import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# -Carrega dados
FILE_PATH = "/home/arthurantunes/Min-de-dados/IA2/archive/world_happiness_processed.csv"
MODEL_FILENAME = "xgb_happiness_model_gscv.joblib"
SCALER_FILENAME = "xgb_happiness_scaler_gscv.joblib"
FEATURES_LIST_FILENAME = "xgb_happiness_features_gscv.joblib"

TARGET_VARIABLE = 'Happiness Score'

try:
    df = pd.read_csv(FILE_PATH)
    print(f"Dados carregados com sucesso de '{FILE_PATH}'!")
except FileNotFoundError:
    print(f"Erro FATAL: Arquivo '{FILE_PATH}' não encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao carregar o arquivo: {e}")
    exit()

# Prepara dados aqui
year_column_candidates = ['Year', 'year', 'Time', 'time', 'Anos', 'ano']
actual_year_column = None
for col_candidate in year_column_candidates:
    if col_candidate in df.columns:
        actual_year_column = col_candidate
        print(f"Coluna de ano encontrada: '{actual_year_column}'")
        break

region_columns = [col for col in df.columns if 'Region_' in col]
numerical_features = [
    'GDP per capita',
    'Social support',
    'Healthy life expectancy',
    'Freedom',
    'Generosity',
    'Perceptions of corruption',
    'Dystopia Residual' if 'Dystopia Residual' in df.columns else None
]
numerical_features = [feat for feat in numerical_features if feat is not None] # Remover nones

features_columns = numerical_features + region_columns
# Remover explicitamentecoluna de ano das features
if actual_year_column and actual_year_column in features_columns:
    print(f"Removendo '{actual_year_column}' das features.")
    features_columns.remove(actual_year_column)
# Remover também Country name
country_name_cols = ['Country name', 'Country or region', 'Country']
for cn_col in country_name_cols:
    if cn_col in features_columns:
        features_columns.remove(cn_col)


missing_feature_cols = [col for col in features_columns if col not in df.columns]
if missing_feature_cols:
    print(f"Erro: As seguintes colunas de features não foram encontradas: {missing_feature_cols}")
    exit()
if TARGET_VARIABLE not in df.columns:
    print(f"Erro: A coluna alvo '{TARGET_VARIABLE}' não foi encontrada.")
    exit()

df_processed = df.dropna(subset=features_columns + [TARGET_VARIABLE]).copy()
if df_processed.empty:
    print("Erro: Após remover NaNs, o DataFrame ficou vazio.")
    exit()

X = df_processed[features_columns]
y = df_processed[TARGET_VARIABLE]
print(f"Features usadas para o treinamento: {X.columns.tolist()}")

#Teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

# Features escalondas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features escalonadas.")

# XGBoost e GridSearchCV
print("\nIniciando GridSearchCV para XGBoost...")
xgb_estimator = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

param_grid = {
    'n_estimators': [100, 200],       
    'max_depth': [3, 5, 7],          
    'learning_rate': [0.05, 0.1],    
    'subsample': [0.8, 0.9],         
    'colsample_bytree': [0.8, 0.9]    
}

grid_search = GridSearchCV(
    estimator=xgb_estimator,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    verbose=1,
    n_jobs=-1 
)

grid_search.fit(X_train_scaled, y_train)

print("\nGridSearchCV concluído.")
print(f"Melhores hiperparâmetros encontrados: {grid_search.best_params_}")
print(f"Melhor pontuação R² na validação cruzada: {grid_search.best_score_:.4f}")


best_xgb_model = grid_search.best_estimator_

# Avalicao
y_pred_test = best_xgb_model.predict(X_test_scaled)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)
print(f"\n--- Desempenho do Melhor XGBoost no Conjunto de Teste ---")
print(f"RMSE: {rmse_test:.4f}")
print(f"R²: {r2_test:.4f}")

# Salvar
joblib.dump(best_xgb_model, MODEL_FILENAME)
joblib.dump(scaler, SCALER_FILENAME)
joblib.dump(features_columns, FEATURES_LIST_FILENAME)
print(f"\nMelhor modelo XGBoost salvo como '{MODEL_FILENAME}'")
print(f"Scaler salvo como '{SCALER_FILENAME}'")
print(f"Lista de features salva como '{FEATURES_LIST_FILENAME}'")
print("\nTreinamento XGBoost com GridSearchCV concluído!")