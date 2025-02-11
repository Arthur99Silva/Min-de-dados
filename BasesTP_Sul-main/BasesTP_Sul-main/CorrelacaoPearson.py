import pandas as pd
from scipy.stats import pearsonr
import numpy as np

# Carregar os datasets
file_path_agriculture = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/Dataset_2016_2024.csv"
file_path_climate = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/ClimaKNN_Filtrado.csv.csv"

df_agriculture = pd.read_csv(file_path_agriculture)
df_climate = pd.read_csv(file_path_climate)

# Converter a coluna de 'Data' para um formato adequado (ano-mês) nos dois datasets
df_agriculture['AnoMes'] = df_agriculture['Data'].astype(str).str.zfill(6).str[-6:]
df_climate['AnoMes'] = df_climate['Data'].astype(str).str.zfill(6).str[-6:]

# Filtrar apenas os dados do estado 1
df_agriculture_state1 = df_agriculture[df_agriculture['Estado'] == 1]
df_climate_state1 = df_climate[df_climate['Estado'] == 1]

# Renomear colunas e converter valores climáticos para numérico
df_climate_state1 = df_climate_state1.rename(columns={
    'TemperaturaMedia C?': 'TemperaturaMedia',
    'PRECIPITACAO TOTAL, MENSAL (AUT)(mm)': 'Precipitacao',
    'VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)': 'VelocidadeVento'
})

df_climate_state1['TemperaturaMedia'] = pd.to_numeric(df_climate_state1['TemperaturaMedia'], errors='coerce')
df_climate_state1['Precipitacao'] = df_climate_state1['Precipitacao'].str.replace(',', '.').astype(float)
df_climate_state1['VelocidadeVento'] = df_climate_state1['VelocidadeVento'].str.replace(',', '.').astype(float)

# Remover dados climáticos com valores nulos
df_climate_state1 = df_climate_state1.dropna(subset=['TemperaturaMedia', 'Precipitacao', 'VelocidadeVento'])

# Lista de cidades únicas no estado 1 com dados climáticos
cities = df_climate_state1['Nome'].unique()

# Lista para armazenar os resultados
correlation_results_all = []

# Função para calcular a correlação de Pearson para um produto específico
def calcular_correlacao(df, produto_nome):
    df_produto = df[df['Produto'] == produto_nome].dropna()
    if len(df_produto) > 1:  # Verificar se há dados suficientes
        try:
            correlation_temp, p_value_temp = pearsonr(df_produto['Produção (t)'], df_produto['TemperaturaMedia'])
            correlation_precip, p_value_precip = pearsonr(df_produto['Produção (t)'], df_produto['Precipitacao'])
            correlation_wind, p_value_wind = pearsonr(df_produto['Produção (t)'], df_produto['VelocidadeVento'])
            correlation_rendimento_temp, p_value_rendimento_temp = pearsonr(df_produto['Rendimento Médio (kg/ha)'], df_produto['TemperaturaMedia'])
            correlation_rendimento_precip, p_value_rendimento_precip = pearsonr(df_produto['Rendimento Médio (kg/ha)'], df_produto['Precipitacao'])
            correlation_rendimento_wind, p_value_rendimento_wind = pearsonr(df_produto['Rendimento Médio (kg/ha)'], df_produto['VelocidadeVento'])
        except ValueError:
            return {}
        return {
            "Produto": produto_nome,
            "Correlação Produção Temperatura": correlation_temp,
            "P-Valor Produção Temperatura": p_value_temp,
            "Correlação Produção Precipitação": correlation_precip,
            "P-Valor Produção Precipitação": p_value_precip,
            "Correlação Produção Velocidade do Vento": correlation_wind,
            "P-Valor Produção Velocidade do Vento": p_value_wind,
            "Correlação Rendimento Temperatura": correlation_rendimento_temp,
            "P-Valor Rendimento Temperatura": p_value_rendimento_temp,
            "Correlação Rendimento Precipitação": correlation_rendimento_precip,
            "P-Valor Rendimento Precipitação": p_value_rendimento_precip,
            "Correlação Rendimento Velocidade do Vento": correlation_rendimento_wind,
            "P-Valor Rendimento Velocidade do Vento": p_value_rendimento_wind,
        }
    else:
        return {}

# Iterar sobre cada cidade
for city in cities:
    df_climate_city = df_climate_state1[df_climate_state1['Nome'] == city]
    df_merged_city = pd.merge(df_agriculture_state1, df_climate_city, on='AnoMes', how='outer')
    df_merged_city = df_merged_city.replace([np.inf, -np.inf], np.nan).dropna()
    if df_merged_city.empty:
        continue
    results_feijao = calcular_correlacao(df_merged_city, 1)
    results_milho = calcular_correlacao(df_merged_city, 2)
    results_trigo = calcular_correlacao(df_merged_city, 3)
    results_feijao["Cidade"] = city
    results_milho["Cidade"] = city
    results_trigo["Cidade"] = city
    correlation_results_all.extend([results_feijao, results_milho, results_trigo])

# Criar DataFrame com os resultados
df_results = pd.DataFrame(correlation_results_all)

# Salvar os resultados em um arquivo TXT
df_results.to_csv("correlacoes_resultados.txt", sep='\t', index=False)

# Exibir os resultados no terminal
print("Resultados das Correlações:")
print(df_results.to_string())

# Selecionar as 10 cidades com mais correlações significativas
top_cities = df_results.groupby("Cidade").size().reset_index(name="Significativo")
top_cities = top_cities.sort_values(by="Significativo", ascending=False).head(10)

df_top_cities = df_results[df_results["Cidade"].isin(top_cities["Cidade"])]

# Exibir os resultados
print("\nTop 10 Cidades com Mais Correlações Significativas:")
print(df_top_cities.to_string())

