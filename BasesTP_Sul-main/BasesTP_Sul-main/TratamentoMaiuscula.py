import pandas as pd
import difflib

# Carregar o arquivo CSV
file_path = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/TEORIADOCARDS.csv"
df = pd.read_csv(file_path)

# Converter os nomes das cidades do arquivo para minúsculas antes da correspondência
df["Nome"] = df["Nome"].astype(str).str.lower()

# Listas de cidades por estado (convertidas para minúsculas)
cidades_por_estado = {
    1: [cidade.lower() for cidade in [
        "Campina da Lagoa", "Castro", "Cidade Gaúcha", "Clevelandia", "Colombo",
        "Dois Vizinhos", "Diamante do Norte", "Foz do Iguaçu", "General Carneiro",
        "Goioerê", "Icaraíma", "Ilha do Mel", "Inácio Martins", "Itapoá", "Ivaí",
        "Japira", "Joaquim Távora", "Laranjeiras do Sul", "Mal. Cândido Rondon",
        "Maringá", "Morretes", "Nova Fátima", "Nova Tebas", "Paranapoema",
        "Rancho Queimado", "São Mateus do Sul", "Ventania", "Curitiba"
    ]],
    2: [cidade.lower() for cidade in [
        "Araranguá", "Bom Jardim da Serra - Morro da Igreja", "Caçador", "Campos Novos", "Chapeco",
        "Curitibanos", "Dionísio Cerqueira", "Florianópolis", "Indaial", "Itajaí",
        "Ituporanga", "Joaçaba", "Lages", "Laguna - Farol de Santa Marta", "Major Vieira",
        "Novo Horizonte", "Rio do Campo", "Rio Negrinho", "São Joaquim",
        "São Miguel do Oeste", "Urussanga", "Xanxerê"
    ]],
    3: [cidade.lower() for cidade in [
        "Alegrete", "Bagé", "Bento Gonçalves", "Cambará do Sul", "Camaquã", "Cangucu",
        "Campo Bom", "Caçapava do Sul", "Canela", "Capão do Leão", "Cruz Alta",
        "Dom Pedrito", "Encruzilhada do Sul", "Erechim", "Frederico Westphalen",
        "Ibirubá", "Jaguarão", "Lagoa Vermelha", "Mostardas", "Palmeira das Missões",
        "Passo Fundo", "Planalto", "Porto Alegre - Belem Novo",
        "Porto Alegre - Jardim Botânico", "Quaraí", "Rancho Queimado", "Rio Grande",
        "Rio Pardo", "Santa Maria", "Santa Rosa", "Santa Vitória do Palmar - Barra do Chuí",
        "Santana do Livramento", "Santiago", "Santo Augusto", "São Borja",
        "São Gabriel", "São José dos Ausentes", "São Luiz Gonzaga",
        "São Vicente do Sul", "Serafina Corrêa", "Soledade", "Teutônia",
        "Torres", "Tramandaí", "Tupanciretã", "Uruguaiana", "Vacaria"
    ]]
}

# Função para encontrar o estado da cidade usando aproximação
def atribuir_estado(cidade):
    for estado, cidades in cidades_por_estado.items():
        melhor_correspondencia = difflib.get_close_matches(cidade, cidades, n=1, cutoff=0.6)
        if melhor_correspondencia:
            return estado
    return None  # Caso não encontre correspondência

# Aplicar a função na coluna "Nome"
df["Estado_Numérico"] = df["Nome"].apply(atribuir_estado)

# Caminho para salvar o novo arquivo
new_file_path = "C:/Users/Arthur/Documents/Documentos/UFSJ/Min de dados/BasesTP_Sul-main/BasesTP_Sul-main/TEORIADOCARDS_Atualizado.csv"

# Salvando o novo arquivo CSV
df.to_csv(new_file_path, index=False)

# Exibir o link para download do novo arquivo
print(f"Arquivo atualizado salvo em: {new_file_path}")
