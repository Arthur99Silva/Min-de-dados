import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb # Necessário para carregar o modelo joblib do XGBoost

# --- 1. Configurações e Carregamento dos Dados ---
FILE_PATH = "/home/arthurantunes/Min-de-dados/IA2/archive/world_happiness_processed.csv" # ATUALIZADO para o novo arquivo
MODEL_FILENAME = "xgb_happiness_model_gscv.joblib" # ATUALIZADO
SCALER_FILENAME = "xgb_happiness_scaler_gscv.joblib" # ATUALIZADO
FEATURES_LIST_FILENAME = "xgb_happiness_features_gscv.joblib" # ATUALIZADO
GRAFICO_RANKING_FILENAME = "ranking_paises_previsto_xgboost_gscv.png" # Nome do gráfico atualizado
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

# --- 2. Carregar o Modelo, o Scaler e a Lista de Features Treinados ---
try:
    xgb_model_loaded = joblib.load(MODEL_FILENAME)
    scaler_loaded = joblib.load(SCALER_FILENAME)
    features_columns_loaded = joblib.load(FEATURES_LIST_FILENAME)
    print(f"Modelo XGBoost '{MODEL_FILENAME}', Scaler '{SCALER_FILENAME}', e Lista de Features '{FEATURES_LIST_FILENAME}' carregados.")
except FileNotFoundError:
    print("Erro FATAL: Arquivo do modelo XGBoost, scaler ou lista de features não encontrado.")
    print("Certifique-se de que o script 'treinar_modelo_xgboost.py' (com GridSearchCV) foi executado primeiro.")
    exit()
except Exception as e:
    print(f"Ocorreu um erro ao carregar modelo/scaler/features: {e}")
    exit()

# --- 3. Preparação dos Dados para Previsão ---
country_column_candidates = ['Country name', 'Country or region', 'Country']
country_column = None
for candidate in country_column_candidates:
    if candidate in df_to_predict.columns:
        country_column = candidate
        break
if country_column is None:
    print("Aviso: Coluna de nome de país não encontrada. Usando índice.")
    df_to_predict['temp_country_id'] = df_to_predict.index
    country_column = 'temp_country_id'

# Verificar se todas as features esperadas pelo modelo estão no DataFrame
# É crucial que features_columns_loaded seja a lista exata usada no treino.
missing_cols = [col for col in features_columns_loaded if col not in df_to_predict.columns]
if missing_cols:
    print(f"Erro: As seguintes colunas de features, esperadas pelo modelo, não foram encontradas no DataFrame: {missing_cols}")
    exit()

df_predict_subset = df_to_predict[[country_column] + features_columns_loaded].copy()
# Tratar NaNs APENAS nas colunas de features que o scaler/modelo espera
df_predict_subset.dropna(subset=features_columns_loaded, inplace=True)

if df_predict_subset.empty:
    print("Erro: Após remover NaNs das features, não há dados para fazer previsões.")
    exit()

X_predict = df_predict_subset[features_columns_loaded]
country_names_for_prediction = df_predict_subset[country_column]

# --- 4. Escalonamento das Features ---
try:
    X_predict_scaled = scaler_loaded.transform(X_predict)
    print("Features para previsão foram escalonadas.")
except ValueError as ve:
    print(f"Erro ao escalar features para previsão: {ve}")
    exit()

# --- 5. Realizar as Previsões ---
predictions = xgb_model_loaded.predict(X_predict_scaled)
print("Previsões realizadas com XGBoost (otimizado com GridSearchCV).")

# --- 6. Apresentar os Resultados ---
df_results = pd.DataFrame({
    'País': country_names_for_prediction.values,
    'Pontuação de Felicidade Prevista (XGBoost_GSCV)': predictions
})

# Adicionar coluna de Score Real se existir no subset (para o print)
if 'Happiness Score' in df_predict_subset.columns:
    df_results['Pontuação de Felicidade Real'] = df_predict_subset['Happiness Score'].values


print("\n--- Previsões da Pontuação de Felicidade (XGBoost com GridSearchCV) ---")
df_results_sorted = df_results.sort_values(by='Pontuação de Felicidade Prevista (XGBoost_GSCV)', ascending=False).reset_index(drop=True)
print(df_results_sorted.head(TOP_N_GRAFICO))

# --- 7. Gerar e Salvar o Gráfico da Classificação ---
# (A lógica de plotagem é a mesma da sua última versão funcional, apenas ajustando os nomes das colunas se necessário)
print("\n--- Preparando dados para o gráfico (XGBoost com GridSearchCV) ---")
df_unique_countries_sorted = df_results_sorted.drop_duplicates(subset=['País'], keep='first')
df_plot = df_unique_countries_sorted.head(TOP_N_GRAFICO)
print(f"Dados para o gráfico (top {len(df_plot)} países únicos):")
print(df_plot)

current_top_n = len(df_plot)
if current_top_n > 0:
    plt.figure(figsize=(12, current_top_n * 0.55))
    sns.set_style("whitegrid")
    barplot = sns.barplot(
        x='Pontuação de Felicidade Prevista (XGBoost_GSCV)', # ATUALIZADO
        y='País',
        data=df_plot,
        palette='crest_r' # Paleta diferente
    )
    plt.title(f'Top {current_top_n} Países (XGBoost_GSCV) por Pontuação Prevista', fontsize=16)
    plt.xlabel('Pontuação de Felicidade Prevista (XGBoost_GSCV)', fontsize=14) # ATUALIZADO
    plt.ylabel('País', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    for index, value in enumerate(df_plot['Pontuação de Felicidade Prevista (XGBoost_GSCV)']): # ATUALIZADO
        barplot.text(value + 0.01, index, f'{value:.2f}', color='black', ha="left", va="center", fontsize=9)
    plt.tight_layout(pad=1.0)
    try:
        plt.savefig(GRAFICO_RANKING_FILENAME)
        print(f"\nGráfico da classificação XGBoost (GSCV) salvo como '{GRAFICO_RANKING_FILENAME}'")
    except Exception as e:
        print(f"Erro ao salvar o gráfico: {e}")
else:
    print("\nNão há dados suficientes (países únicos) para gerar o gráfico.")

print("\nPrevisão XGBoost (GSCV) e geração de gráfico concluídas!")