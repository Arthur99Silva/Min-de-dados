import pandas as pd

# Carregar o arquivo CSV
file_path = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/New_TempSul.csv"
df = pd.read_csv(file_path)

# Definir as datas que devem ser mantidas conforme o script Pearson.py
datas_validas = {
    12016, 22016, 32016, 42016, 52016, 62016, 72016, 82016, 92016, 102016, 112016, 122016,
    12017, 22017, 32017, 42017, 52017, 62017, 72017, 82017, 92017, 102017, 112017, 122017,
    12018, 22018, 32018, 42018, 52018, 62018, 72018, 82018, 92018, 102018, 112018, 122018,
    12019, 22019, 32019, 42019, 52019, 62019, 72019, 82019, 92019, 102019, 112019, 122019,
    12020, 22020, 32020, 42020, 52020, 62020, 72020, 82020, 92020, 102020, 112020, 122020,
    12021, 22021, 32021, 42021, 52021, 62021, 72021, 82021, 92021, 102021, 112021, 122021,
    12022, 22022, 32022, 42022, 52022, 62022, 72022, 82022, 92022, 102022, 112022, 122022,
    12023, 22023, 32023, 42023, 52023, 62023, 72023, 82023, 92023, 102023, 112023, 122023,
    12024, 22024, 32024, 42024, 52024, 62024, 72024, 82024, 92024, 102024, 112024, 122024
}

# Filtrar o dataframe mantendo apenas as datas desejadas
df_filtrado = df[df["Data"].isin(datas_validas)]

# Salvar o novo arquivo CSV
output_path = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/ClimaSul.csv"
df_filtrado.to_csv(output_path, index=False)
