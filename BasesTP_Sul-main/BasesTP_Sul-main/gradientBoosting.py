import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')

# Carregar os datasets
clima_df = pd.read_csv("/home/arthurantunes/Documentos/Min-de-dados-main/BasesTP_Sul-main/BasesTP_Sul-main/ClimaSul.csv")
agricola_df = pd.read_csv("//home/arthurantunes/Documentos/Min-de-dados-main/BasesTP_Sul-main/BasesTP_Sul-main/Dataset_2016_2024.csv")

# Converter colunas numéricas que possuem vírgulas para ponto decimal no dataset climático
for col in ["PRECIPITACAO TOTAL, MENSAL (AUT)(mm)", "PRESSAO ATMOSFERICA, MEDIA MENSAL (AUT)(mB)", 
            "VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)"]:
    clima_df[col] = clima_df[col].astype(str).str.replace(',', '.').astype(float)

# Remover espaços extras nos nomes das colunas
clima_df.columns = clima_df.columns.str.strip()

# Renomear colunas para facilitar o uso no modelo
clima_df.rename(columns={
    "TemperaturaMedia C": "TemperaturaMedia",
    "PRECIPITACAO TOTAL, MENSAL (AUT)(mm)": "PrecipitacaoMensal",
    "PRESSAO ATMOSFERICA, MEDIA MENSAL (AUT)(mB)": "PressaoAtmosferica",
    "VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)": "VelocidadeVento"
}, inplace=True)

# Mesclar os dados com base em Data e Estado
merged_df = pd.merge(agricola_df, clima_df, on=["Data", "Estado"], how="inner")

# Remover colunas desnecessárias
merged_df.drop(columns=["Cidade"], inplace=True, errors='ignore')

# Remover linhas com valores nulos
merged_df = merged_df.dropna()

# Definir as features (X) e o alvo (y)
X = merged_df[['Produto', 'Estado', 'Rendimento Médio (kg/ha)', 
               'TemperaturaMedia', 'PrecipitacaoMensal', 'PressaoAtmosferica', 'VelocidadeVento']]

y = merged_df['Área Colhida (ha)']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar um modelo Gradient Boosting com ajuste de hiperparâmetros
gb_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

# Fazer previsões
y_pred_gb = gb_model.predict(X_test)

# Avaliação do modelo Gradient Boosting
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# Impressão dos resultados
print("Gradient Boosting (Ajustado):")
print(f"MAE: {mae_gb}, MSE: {mse_gb}, RMSE: {rmse_gb}, R²: {r2_gb}")

# Criar gráficos para visualização dos resultados
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred_gb)
plt.xlabel("Valores Reais")
plt.ylabel("Previsões")
plt.title("Valores Reais vs Previsões (Gradient Boosting)")
plt.savefig("scatterplot_gb.png")  # Salvar gráfico em arquivo

# Gráfico de erros
errors = y_test - y_pred_gb
plt.figure(figsize=(10, 5))
sns.histplot(errors, bins=30, kde=True)
plt.xlabel("Erro")
plt.ylabel("Frequência")
plt.title("Distribuição dos Erros (Gradient Boosting)")
plt.savefig("erro_histograma_gb.png")

# Importância das Features
importances = gb_model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(12, 6))
sns.barplot(x=importances, y=feature_names)
plt.xlabel("Importância")
plt.ylabel("Variáveis")
plt.title("Importância das Features no Modelo (Gradient Boosting)")
plt.savefig("importancia_features_gb.png")
