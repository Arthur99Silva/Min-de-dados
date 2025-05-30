import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler # Mantido para consistência, embora RF seja menos sensível
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv('IA2/archive/world_happiness_processed.csv')
    print("Dados carregados com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivo 'world_happiness_processed.csv' não encontrado.")
    print("Por favor, certifique-se de que o arquivo está no mesmo diretório do script ou forneça o caminho correto.")
    exit()

target_variable = 'Happiness Score'

numerical_features = [
    'GDP per capita',
    'Social support',
    'Healthy life expectancy',
    'Freedom',
    'Generosity',
    'Perceptions of corruption'
]
features_columns = numerical_features
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nFormato de X_train: {X_train.shape}")
print(f"Formato de X_test: {X_test.shape}")
print(f"Formato de y_train: {y_train.shape}")
print(f"Formato de y_test: {y_test.shape}")


scaler = StandardScaler()
X_train_numerical_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_numerical_scaled = scaler.transform(X_test[numerical_features])

X_train_numerical_scaled_df = pd.DataFrame(X_train_numerical_scaled, columns=numerical_features, index=X_train.index)
X_test_numerical_scaled_df = pd.DataFrame(X_test_numerical_scaled, columns=numerical_features, index=X_test.index)

X_train_scaled = pd.concat([X_train_numerical_scaled_df, X_train[numerical_features]], axis=1) #Alterado
X_test_scaled = pd.concat([X_test_numerical_scaled_df, X_test[numerical_features]], axis=1) #Alterado

print("\nFeatures numéricas padronizadas (para consistência).")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
rf_model.fit(X_train_scaled, y_train)

print("\nModelo Random Forest Regressor treinado.")
print(f"Pontuação OOB (Out-of-Bag) do modelo: {rf_model.oob_score_:.4f}")
print("(A pontuação OOB é uma estimativa do desempenho do modelo em dados não vistos, calculada durante o treinamento)")

y_pred_rf = rf_model.predict(X_test_scaled)

mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n--- Métricas de Avaliação para Random Forest Regressor ---")
print(f"Erro Quadrático Médio (MSE): {mse_rf:.4f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse_rf:.4f}")
print(f"Erro Absoluto Médio (MAE): {mae_rf:.4f}")
print(f"Coeficiente de Determinação (R²): {r2_rf:.4f}")

# r2
if r2_rf > 0.7:
    print("O modelo Random Forest explica uma boa parte da variância do Happiness Score.")
elif r2_rf > 0.5:
    print("O modelo Random Forest tem um poder explicativo moderado sobre o Happiness Score.")
else:
    print("O modelo Random Forest tem um poder explicativo baixo sobre o Happiness Score.")

# Feature mais importantes
importances = rf_model.feature_importances_
feature_names = X_train_scaled.columns
forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("\n--- Importância das Features (Random Forest) ---")
print(forest_importances)

plt.figure(figsize=(12, 8))
sns.barplot(x=forest_importances.values, y=forest_importances.index, palette="viridis")
plt.title('Importância das Features - Random Forest Regressor')
plt.xlabel('Importância Relativa')
plt.ylabel('Features')
plt.tight_layout()
print("\nGráfico de importância das features gerado (pode precisar de plt.show() para exibir).")
plt.savefig("grafico.png")