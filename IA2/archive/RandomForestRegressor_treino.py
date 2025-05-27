import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# Carregamento dos Dados 
FILE_PATH = "IA2/archive/world_happiness_processed.csv"
MODEL_FILENAME = "rf_happiness_model.joblib"
SCALER_FILENAME = "happiness_scaler.joblib"
FEATURES_LIST_FILENAME = "happiness_features.joblib"

TARGET_VARIABLE = 'Happiness Score'

try:
    df = pd.read_csv(FILE_PATH)
    print(f"Dados carregados com sucesso de '{FILE_PATH}'!")
except FileNotFoundError:
    print(f"Erro FATAL: Arquivo '{FILE_PATH}' não encontrado.")
    print("Por favor, certifique-se de que o arquivo está no local correto.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao carregar o arquivo: {e}")
    exit()

# Identificar colunas as features
region_columns = [col for col in df.columns if 'Region_' in col]
numerical_features = [
    'GDP per capita',
    'Social support',
    'Healthy life expectancy',
    'Freedom',
    'Generosity',
    'Perceptions of corruption'
]
features_columns = numerical_features + region_columns

missing_feature_cols = [col for col in features_columns if col not in df.columns]
if missing_feature_cols:
    print(f"Erro: As seguintes colunas de features não foram encontradas: {missing_feature_cols}")
    exit()
if TARGET_VARIABLE not in df.columns:
    print(f"Erro: A coluna alvo '{TARGET_VARIABLE}' não foi encontrada.")
    exit()

# Remover linhas onde as features OU o alvo são NaN
df_processed = df.dropna(subset=features_columns + [TARGET_VARIABLE]).copy()

if df_processed.empty:
    print("Erro: Após remover NaNs, o DataFrame ficou vazio. Não é possível treinar.")
    exit()

X = df_processed[features_columns]
y = df_processed[TARGET_VARIABLE]

# reino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

# escalonamento das Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features escalonadas.")

# RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
rf_model.fit(X_train_scaled, y_train)
print(f"Modelo Random Forest treinado. Pontuação OOB: {rf_model.oob_score_:.4f}")

# Avaliação do Modelo
y_pred_test = rf_model.predict(X_test_scaled)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)
print(f"\n--- Desempenho no Conjunto de Teste ---")
print(f"RMSE: {rmse_test:.4f}")
print(f"R²: {r2_test:.4f}")

# Salvamento
joblib.dump(rf_model, MODEL_FILENAME)
joblib.dump(scaler, SCALER_FILENAME)
joblib.dump(features_columns, FEATURES_LIST_FILENAME)
print(f"\nModelo salvo como '{MODEL_FILENAME}'")
print(f"Scaler salvo como '{SCALER_FILENAME}'")
print(f"Lista de features salva como '{FEATURES_LIST_FILENAME}'")
print("\nTreinamento concluído!")