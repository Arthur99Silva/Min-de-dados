import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuração para evitar erro do Matplotlib
matplotlib.use('Agg')

# Carregar os dados
file_path = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/Dataset_Filtrado.csv"
df = pd.read_csv(file_path)

# Remover colunas irrelevantes e valores NaN
df_clean = df.drop(columns=['Data'], errors='ignore').dropna()

# Análise de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df_clean.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Matriz de Correlação")
plt.savefig("correlacao_heatmap_hibrido.png")
print("\nO gráfico de correlação foi salvo como 'correlacao_heatmap_hibrido.png'.")

# Remover outliers
q1 = df_clean['Produção (t)'].quantile(0.25)
q3 = df_clean['Produção (t)'].quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

df_clean = df_clean[(df_clean['Produção (t)'] >= limite_inferior) & (df_clean['Produção (t)'] <= limite_superior)]

# Definir variáveis independentes e dependente
X = df_clean.drop(columns=['Produção (t)'])
y = df_clean['Produção (t)']

# Separação dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)

# Configuração dos hiperparâmetros otimizados
rf_params = {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
gbr_params = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.9}

# Treinar os modelos base
rf_model = RandomForestRegressor(**rf_params, random_state=42)
gbr_model = GradientBoostingRegressor(**gbr_params, random_state=42)

rf_model.fit(X_train, y_train)
gbr_model.fit(X_train, y_train)

# Fazer previsões individuais dos modelos
rf_pred = rf_model.predict(X_test)
gbr_pred = gbr_model.predict(X_test)

# Criar um novo dataset com as previsões dos dois modelos
stacked_features = np.column_stack((rf_pred, gbr_pred))

# Treinar um modelo final para combinar as previsões
meta_model = LinearRegression()
meta_model.fit(stacked_features, y_test)

# Fazer previsões finais usando o modelo híbrido
final_predictions = meta_model.predict(stacked_features)

# Avaliação do modelo híbrido
mae = mean_absolute_error(y_test, final_predictions)
mse = mean_squared_error(y_test, final_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, final_predictions)

print("\n--- Resultados do Modelo Híbrido ---")
print(f"Erro Absoluto Médio (MAE): {mae:.2f}")
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.5f}")

# Importância das variáveis baseada no modelo de Random Forest
importances = rf_model.feature_importances_
features = X.columns
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances[sorted_indices], align="center")
plt.xticks(range(len(importances)), np.array(features)[sorted_indices], rotation=90)
plt.ylabel("Importância")
plt.title("Importância das Variáveis no Modelo Híbrido")
plt.savefig("importancia_variaveis_hibrido.png")
print("\nO gráfico de importância das variáveis foi salvo como 'importancia_variaveis_hibrido.png'.")
