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

# Escolher a primeira cidade com todos os dados climáticos disponíveis
df_climate_state1 = df_climate_state1.dropna(subset=['TemperaturaMedia', 'Precipitacao', 'VelocidadeVento'])
first_city = df_climate_state1['Nome'].unique()[0]

df_climate_city = df_climate_state1[df_climate_state1['Nome'] == first_city]

# Mesclar os datasets pelo AnoMes
df_merged = pd.merge(df_agriculture_state1, df_climate_city, on='AnoMes', how='inner')

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

# **MODIFIQUE AQUI PARA TESTAR OUTRAS CULTURAS**
# Produto 1 = Feijão, Produto 2 = Milho, Produto 3 = Trigo
correlation_results_feijao = calcular_correlacao(df_merged, 1)
correlation_results_milho = calcular_correlacao(df_merged, 2)
correlation_results_trigo = calcular_correlacao(df_merged, 3)

# Exibir resultados
correlation_results_feijao, correlation_results_milho, correlation_results_trigo
