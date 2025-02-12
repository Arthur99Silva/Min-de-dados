import pandas as pd
import matplotlib.pyplot as plt

# Carregar a tabela do CSV
comparacao_df = pd.read_csv("/home/arthurantunes/Documentos/Min-de-dados-main/BasesTP_Sul-main/BasesTP_Sul-main/comparacao_area_colhida.csv")

# Criar figura e eixo
fig, ax = plt.subplots(figsize=(12, 6))

# Esconder os eixos
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

# Criar tabela na imagem
table = ax.table(cellText=comparacao_df.values,
                 colLabels=comparacao_df.columns,
                 cellLoc='center',
                 loc='center')

# Ajustar layout
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([i for i in range(len(comparacao_df.columns))])

# Salvar a imagem
plt.savefig("comparacao_modelos_prod.png", dpi=300, bbox_inches='tight')

print("Imagem 'comparacao_modelos.png' salva com sucesso!")
