import pandas as pd
from scipy.stats import pearsonr

# Carregar os arquivos CSV
clima_path = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/ClimaSul.csv"
agricultura_path = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/Dataset_2016_2024.csv"

df_clima = pd.read_csv(clima_path)
df_agricultura = pd.read_csv(agricultura_path)

# Converter colunas numéricas com valores errados de string para float
df_clima["PRECIPITACAO TOTAL, MENSAL (AUT)(mm)"] = (
    df_clima["PRECIPITACAO TOTAL, MENSAL (AUT)(mm)"]
    .astype(str)
    .str.replace(",", ".")
    .astype(float)
)
df_clima["VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)"] = (
    df_clima["VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)"]
    .astype(str)
    .str.replace(",", ".")
    .astype(float)
)

# Filtrar apenas o Estado 1
df_clima_estado1 = df_clima[df_clima["Estado"] == 1]
df_agricultura_estado1 = df_agricultura[df_agricultura["Estado"] == 1]

# Identificar cidades com registros completos de temperatura, precipitação e vento
cidades_validas = df_clima_estado1.dropna(
    subset=[
        "TemperaturaMedia C",
        "PRECIPITACAO TOTAL, MENSAL (AUT)(mm)",
        "VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)",
    ]
)["Cidade"].unique()

# Lista para armazenar os resultados de todas as cidades
resultados_finais = []

# Lista de produtos a serem analisados
produtos_analise = [1, 2, 5, 7]

# Iterar por todas as cidades válidas no Estado 1
for cidade in cidades_validas:
    df_clima_cidade = df_clima_estado1[df_clima_estado1["Cidade"] == cidade]
    df_clima_cidade["Data"] = df_clima_cidade["Data"].astype(str)

    # Mesclar os datasets pelo campo Data
    df_merged = pd.merge(df_agricultura_estado1, df_clima_cidade, on="Data", how="inner")

    # Aplicar a correlação de Pearson para os produtos 1, 2, 5 e 7 na cidade
    correlacoes = []
    for produto in produtos_analise:
        df_filtrado = df_merged[df_merged["Produto"] == produto].dropna(
            subset=["TemperaturaMedia C", "PRECIPITACAO TOTAL, MENSAL (AUT)(mm)", "VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)", "Rendimento Médio (kg/ha)"]
        )

        if not df_filtrado.empty:
            for variavel in ["TemperaturaMedia C", "PRECIPITACAO TOTAL, MENSAL (AUT)(mm)", "VENTO, VELOCIDADE MEDIA MENSAL (AUT)(m/s)"]:
                corr, p_value = pearsonr(df_filtrado[variavel], df_filtrado["Rendimento Médio (kg/ha)"])
                correlacoes.append({"Cidade": cidade, "Produto": produto, "Variável Climática": variavel, "Correlação": corr, "P-Valor": p_value})

    # Adicionar os resultados ao conjunto final
    resultados_finais.extend(correlacoes)

# Criar DataFrame com os resultados de todas as cidades
df_resultados_finais = pd.DataFrame(resultados_finais)

# Filtrar apenas os resultados com p-valor < 0.05 (significativos)
df_significativos = df_resultados_finais[df_resultados_finais["P-Valor"] < 0.05]

df_significativos
