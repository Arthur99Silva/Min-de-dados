import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuração para evitar erro do Matplotlib em alguns ambientes
matplotlib.use('Agg')

# Carregar os dados
file_path = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/Completo.csv"
df = pd.read_csv(file_path)

# Remover colunas irrelevantes e valores NaN
df_clean = df.drop(columns=['Data'], errors='ignore').dropna()

# Análise de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df_clean.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Matriz de Correlação")
plt.savefig("correlacao_heatmap_gbr.png")
print("\nO gráfico de correlação foi salvo como 'correlacao_heatmap_gbr.png'.")

# Remover outliers
q1 = df_clean['Rendimento Médio (kg/ha)'].quantile(0.25)
q3 = df_clean['Rendimento Médio (kg/ha)'].quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

df_clean = df_clean[(df_clean['Rendimento Médio (kg/ha)'] >= limite_inferior) & (df_clean['Rendimento Médio (kg/ha)'] <= limite_superior)]

# Definir variáveis independentes e dependente
X = df_clean.drop(columns=['Rendimento Médio (kg/ha)'])
y = df_clean['Rendimento Médio (kg/ha)']

# Separação dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)

# Definir hiperparâmetros ajustados para Gradient Boosting
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 10],
    'subsample': [0.7, 0.9, 1.0]
}

gbr = GradientBoostingRegressor(random_state=42)

# Grid Search
grid_search = GridSearchCV(gbr, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Resultados Otimizados Gradient Boosting ---")
print(f"Melhores Hiperparâmetros: {grid_search.best_params_}")
print(f"Erro Absoluto Médio (MAE): {mae:.2f}")
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.5f}")

# Importância das variáveis
importances = best_model.feature_importances_
features = X.columns
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances[sorted_indices], align="center")
plt.xticks(range(len(importances)), np.array(features)[sorted_indices], rotation=90)
plt.ylabel("Importância")
plt.title("Importância das Variáveis no Modelo Gradient Boosting")
plt.savefig("importancia_variaveis_gbr.png")
print("\nO gráfico de importância das variáveis foi salvo como 'importancia_variaveis_gbr.png'.")
