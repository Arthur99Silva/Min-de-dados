import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Carregar dados
FILE_PATH = "IA2/archive/world_happiness_processed.csv"
MODEL_FILENAME = "xgb_happiness_model.joblib"
SCALER_FILENAME = "xgb_happiness_scaler.joblib"
FEATURES_LIST_FILENAME = "xgb_happiness_features.joblib"
GRAFICO_RANKING_FILENAME = "ranking_paises_previsto_xgboost.png"
TOP_N_GRAFICO = 20

try:
    df_to_predict = pd.read_csv(FILE_PATH)
    print(f"Dados para previsão carregados de '{FILE_PATH}'!")
except FileNotFoundError:
    print(f"Erro FATAL: Arquivo '{FILE_PATH}' não encontrado.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao carregar o arquivo de dados: {e}")
    exit()

# Carrgar modelo treinado
try:
    xgb_model_loaded = joblib.load(MODEL_FILENAME)
    scaler_loaded = joblib.load(SCALER_FILENAME)
    features_columns_loaded = joblib.load(FEATURES_LIST_FILENAME)
    print(f"Modelo XGBoost '{MODEL_FILENAME}', Scaler '{SCALER_FILENAME}', e Lista de Features '{FEATURES_LIST_FILENAME}' carregados.")
except FileNotFoundError:
    print("Erro FATAL: Arquivo do modelo XGBoost, scaler ou lista de features não encontrado.")
    print("Certifique-se de que o script 'treinar_modelo_xgboost.py' foi executado primeiro.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao carregar modelo/scaler/features: {e}")
    exit()

# Prpara dados
country_column_candidates = ['Country name', 'Country or region', 'Country']
country_column = None
for candidate in country_column_candidates:
    if candidate in df_to_predict.columns:
        country_column = candidate
        break
if country_column is None:
    df_to_predict['temp_country_id'] = df_to_predict.index
    country_column = 'temp_country_id'

missing_cols = [col for col in features_columns_loaded if col not in df_to_predict.columns]
if missing_cols:
    print(f"Erro: Features esperadas pelo modelo não encontradas no DataFrame: {missing_cols}")
    exit()

df_predict_subset = df_to_predict[[country_column] + features_columns_loaded].copy()
df_predict_subset.dropna(subset=features_columns_loaded, inplace=True)
if df_predict_subset.empty:
    print("Erro: Não há dados para fazer previsões após remover NaNs das features.")
    exit()

X_predict = df_predict_subset[features_columns_loaded]
country_names_for_prediction = df_predict_subset[country_column]

# escalonamento das Features
try:
    X_predict_scaled = scaler_loaded.transform(X_predict)
    print("Features para previsão foram escalonadas.")
except ValueError as ve:
    print(f"Erro ao escalar features para previsão: {ve}")
    exit()

# Previsão
predictions = xgb_model_loaded.predict(X_predict_scaled)
print("Previsões realizadas com XGBoost.")

# Resultados
df_results = pd.DataFrame({
    'País': country_names_for_prediction.values,
    'Pontuação de Felicidade Prevista (XGBoost)': predictions
})

if 'Happiness Score' in df_predict_subset.columns:
    df_results['Pontuação de Felicidade Real'] = df_predict_subset['Happiness Score'].values

print("\n--- Previsões da Pontuação de Felicidade (XGBoost) ---")
df_results_sorted = df_results.sort_values(by='Pontuação de Felicidade Prevista (XGBoost)', ascending=False).reset_index(drop=True)
print(df_results_sorted.head(TOP_N_GRAFICO))

# Salvar
print("\n--- Preparando dados para o gráfico (XGBoost) ---")
df_unique_countries_sorted = df_results_sorted.drop_duplicates(subset=['País'], keep='first')
df_plot = df_unique_countries_sorted.head(TOP_N_GRAFICO)
print(f"Dados para o gráfico (top {len(df_plot)} países únicos):")
print(df_plot)

current_top_n = len(df_plot)
if current_top_n > 0:
    plt.figure(figsize=(12, current_top_n * 0.55))
    sns.set_style("whitegrid")
    barplot = sns.barplot(
        x='Pontuação de Felicidade Prevista (XGBoost)',
        y='País',
        data=df_plot,
        palette='mako_r'
    )
    plt.title(f'Top {current_top_n} Países (XGBoost) por Pontuação de Felicidade Prevista', fontsize=16)
    plt.xlabel('Pontuação de Felicidade Prevista (XGBoost)', fontsize=14)
    plt.ylabel('País', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    for index, value in enumerate(df_plot['Pontuação de Felicidade Prevista (XGBoost)']):
        barplot.text(value + 0.01, index, f'{value:.2f}', color='black', ha="left", va="center", fontsize=9)
    plt.tight_layout(pad=1.0)
    try:
        plt.savefig(GRAFICO_RANKING_FILENAME)
        print(f"\nGráfico da classificação XGBoost salvo como '{GRAFICO_RANKING_FILENAME}'")
    except Exception as e:
        print(f"Erro ao salvar o gráfico: {e}")
else:
    print("\nNão há dados suficientes (países únicos) para gerar o gráfico.")

print("\nPrevisão XGBoost e geração de gráfico concluídas!")