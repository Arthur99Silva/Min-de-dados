import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 1. Carregamento dos Dados
try:
    df = pd.read_csv('world_happiness_processed.csv')
    print("Dados carregados com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivo 'world_happiness_processed.csv' não encontrado.")
    print("Por favor, certifique-se de que o arquivo está no mesmo diretório do script ou forneça o caminho correto.")
    exit()

# 2. Seleção da Variável Alvo (Target) e Variáveis Explicativas (Features)
target_variable = 'Happiness Score'

# Features selecionadas conforme o planejamento
# Incluindo as colunas one-hot encoded para 'Region'
# Primeiro, vamos identificar todas as colunas de região
region_columns = [col for col in df.columns if 'Region_' in col]

# Features numéricas principais
numerical_features = [
    'GDP per capita',
    'Social support',
    'Healthy life expectancy',
    'Freedom',
    'Generosity',
    'Perceptions of corruption'
]

# Combinar features numéricas com as colunas de região
features_columns = numerical_features + region_columns

# Verificar se todas as colunas de features existem no DataFrame
missing_cols = [col for col in features_columns if col not in df.columns]
if missing_cols:
    print(f"Erro: As seguintes colunas de features não foram encontradas no DataFrame: {missing_cols}")
    exit()

if target_variable not in df.columns:
    print(f"Erro: A coluna alvo '{target_variable}' não foi encontrada no DataFrame.")
    exit()

X = df[features_columns]
y = df[target_variable]

print(f"\nVariável Alvo: {target_variable}")
print(f"Features Selecionadas ({len(features_columns)}): {features_columns}")

# 3. Divisão dos Dados em Conjuntos de Treinamento e Teste
# 80% para treino, 20% para teste. random_state para reprodutibilidade.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nFormato de X_train: {X_train.shape}")
print(f"Formato de X_test: {X_test.shape}")
print(f"Formato de y_train: {y_train.shape}")
print(f"Formato de y_test: {y_test.shape}")

# 4. Normalização/Padronização das Features
# A Regressão Linear pode se beneficiar da padronização, especialmente se as features tiverem escalas muito diferentes.
# Vamos padronizar apenas as features numéricas. As colunas de região já são 0 ou 1.

scaler = StandardScaler()

# Ajustar o scaler APENAS nos dados de treinamento das features numéricas
X_train_numerical_scaled = scaler.fit_transform(X_train[numerical_features])
# Aplicar a mesma transformação aos dados de teste das features numéricas
X_test_numerical_scaled = scaler.transform(X_test[numerical_features])

# Converter de volta para DataFrames do Pandas para manter os nomes das colunas
X_train_numerical_scaled_df = pd.DataFrame(X_train_numerical_scaled, columns=numerical_features, index=X_train.index)
X_test_numerical_scaled_df = pd.DataFrame(X_test_numerical_scaled, columns=numerical_features, index=X_test.index)

# Combinar as features numéricas padronizadas com as features de região (que não foram escalonadas)
X_train_scaled = pd.concat([X_train_numerical_scaled_df, X_train[region_columns]], axis=1)
X_test_scaled = pd.concat([X_test_numerical_scaled_df, X_test[region_columns]], axis=1)

print("\nFeatures numéricas padronizadas.")
print("Primeiras linhas de X_train_scaled:")
print(X_train_scaled.head())

# 5. Instanciação e Treinamento do Modelo de Regressão Linear
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

print("\nModelo de Regressão Linear treinado.")

# (Opcional) Visualizar os coeficientes do modelo
# print("\nCoeficientes do modelo (intercepto e pesos das features):")
# print(f"Intercepto (beta_0): {linear_model.intercept_}")
# coefficients_df = pd.DataFrame(linear_model.coef_, X_train_scaled.columns, columns=['Coeficiente'])
# print(coefficients_df.sort_values(by='Coeficiente', ascending=False))


# 6. Realização de Previsões no Conjunto de Teste
y_pred_linear = linear_model.predict(X_test_scaled)

# 7. Análise dos Resultados - Métricas de Avaliação
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear) # Ou mean_squared_error(y_test, y_pred_linear, squared=False)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("\n--- Métricas de Avaliação para Regressão Linear ---")
print(f"Erro Quadrático Médio (MSE): {mse_linear:.4f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse_linear:.4f}")
print(f"Erro Absoluto Médio (MAE): {mae_linear:.4f}")
print(f"Coeficiente de Determinação (R²): {r2_linear:.4f}")

# Interpretação básica do R²:
if r2_linear > 0.7:
    print("O modelo de Regressão Linear explica uma boa parte da variância do Happiness Score.")
elif r2_linear > 0.5:
    print("O modelo de Regressão Linear tem um poder explicativo moderado sobre o Happiness Score.")
else:
    print("O modelo de Regressão Linear tem um poder explicativo baixo sobre o Happiness Score.")

# Exemplo de como você poderia apresentar os resultados no seu relatório:
# "O modelo de Regressão Linear alcançou um RMSE de [valor_rmse] e um R² de [valor_r2] no conjunto de teste.
# Isso indica que, em média, as previsões do modelo desviam em [valor_rmse] unidades do Happiness Score real,
# e o modelo consegue explicar [valor_r2 * 100]% da variância na pontuação de felicidade."

