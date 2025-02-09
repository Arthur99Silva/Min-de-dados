import pandas as pd
from scipy.stats import pearsonr

# Carregar os datasets
file_path_agriculture = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/Numerico_Prime.csv"
file_path_climate = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/TEORIADOCARDS_Atualizado.csv"


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
    df_produto = df[df['Produto'] == produto_nome]
    if len(df_produto) > 1:  # Verificar se há dados suficientes
        correlation_temp, p_value_temp = pearsonr(df_produto['Produção (t)'], df_produto['TemperaturaMedia'])
        correlation_precip, p_value_precip = pearsonr(df_produto['Produção (t)'], df_produto['Precipitacao'])
        correlation_wind, p_value_wind = pearsonr(df_produto['Produção (t)'], df_produto['VelocidadeVento'])
        return {
            "Produto": produto_nome,
            "Correlação Temperatura": correlation_temp,
            "P-Valor Temperatura": p_value_temp,
            "Correlação Precipitação": correlation_precip,
            "P-Valor Precipitação": p_value_precip,
            "Correlação Velocidade do Vento": correlation_wind,
            "P-Valor Velocidade do Vento": p_value_wind,
        }
    else:
        return {
            "Produto": produto_nome,
            "Correlação Temperatura": None,
            "P-Valor Temperatura": None,
            "Correlação Precipitação": None,
            "P-Valor Precipitação": None,
            "Correlação Velocidade do Vento": None,
            "P-Valor Velocidade do Vento": None,
        }

# Iterar sobre cada cidade
for city in cities:
    df_climate_city = df_climate_state1[df_climate_state1['Nome'] == city]
    df_merged_city = pd.merge(df_agriculture_state1, df_climate_city, on='AnoMes', how='inner')
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

# Contar correlações significativas por cidade
df_results['Significativo'] = (
    (df_results['P-Valor Temperatura'] < 0.05) |
    (df_results['P-Valor Precipitação'] < 0.05) |
    (df_results['P-Valor Velocidade do Vento'] < 0.05)
)

# Selecionar as 10 cidades com mais correlações significativas
top_cities = df_results.groupby("Cidade")["Significativo"].sum().reset_index()
top_cities = top_cities.sort_values(by="Significativo", ascending=False).head(10)

df_top_cities = df_results[df_results["Cidade"].isin(top_cities["Cidade"])]

# Exibir os resultados
df_top_cities
