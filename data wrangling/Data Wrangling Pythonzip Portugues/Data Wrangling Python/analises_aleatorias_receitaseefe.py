# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:55:51 2024

@author: Amilcar
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração do estilo dos gráficos
sns.set(style='whitegrid')

# Convertendo a coluna 'Data' para datetime e extraindo o ano
receitas['Data'] = pd.to_datetime(receitas['Data'])
receitas['Ano'] = receitas['Data'].dt.year

# Garantindo que todos os valores na coluna 'Valor' sejam strings
receitas['Valor'] = receitas['Valor'].astype(str)

# Limpando e convertendo a coluna 'Valor' de string para float
receitas['Valor'] = receitas['Valor'].str.replace('.', '', regex=False)  # Remove pontos de separação de milhares
receitas['Valor'] = receitas['Valor'].str.replace(',', '.', regex=False)  # Substitui vírgula por ponto decimal
receitas['Valor'] = pd.to_numeric(receitas['Valor'], errors='coerce')  # Converte para float, tratando erros

# Verificando se há valores NaN após a conversão e, se houver, pode-se filtrar ou tratar de acordo
receitas = receitas.dropna(subset=['Valor'])  # Remove linhas com NaN na coluna 'Valor' (opcional)

# Criando faixas de valores
bins = [0, 1000, 5000, 10000, 50000, 100000, receitas['Valor'].max()]
labels = ['0-1K', '1K-5K', '5K-10K', '10K-50K', '50K-100K', '100K+']
receitas['Faixa de Valor'] = pd.cut(receitas['Valor'], bins=bins, labels=labels)

# Filtrando fontes de recurso com valores totais significativos
valor_por_fonte = receitas.groupby('Fonte Recurso')['Valor'].sum()
fontes_significativas = valor_por_fonte[valor_por_fonte > valor_por_fonte.quantile(0.1)].index
receitas_filtradas = receitas[receitas['Fonte Recurso'].isin(fontes_significativas)]

# Filtrando apenas os anos de interesse
anos_interesse = [2010, 2014, 2018, 2022]
receitas_filtradas = receitas_filtradas[receitas_filtradas['Ano'].isin(anos_interesse)]

# Gráfico de linha para sazonalidade com faixas de valores
plt.figure(figsize=(14, 6))
sns.lineplot(data=receitas_filtradas, x='Ano', y='Valor', hue='Fonte Recurso')
plt.title('Sazonalidade das Receitas ao Longo dos Anos')
plt.xlabel('Ano')
plt.ylabel('Valor (por faixas)')
plt.xticks(anos_interesse)  # Ajustando o eixo x para mostrar apenas os anos desejados
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração do estilo dos gráficos
sns.set(style='whitegrid')

# Selecionando apenas as principais fontes de recurso para o gráfico
fontes_principais = receitas_filtradas['Fonte Recurso'].value_counts().head(5).index
receitas_filtradas = receitas_filtradas[receitas_filtradas['Fonte Recurso'].isin(fontes_principais)]

# Gráfico de linha para sazonalidade com datas completas
plt.figure(figsize=(14, 6))
sns.lineplot(data=receitas_filtradas, x='Data', y='Valor', hue='Fonte Recurso')

# Adicionando anotações
plt.title('Sazonalidade das Receitas ao Longo dos Anos (Principais Fontes)')
plt.xlabel('Data')
plt.ylabel('Valor (por faixas)')
plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo x para melhor legibilidade
plt.tight_layout()  # Ajusta o layout para evitar sobreposição

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração do estilo dos gráficos
sns.set(style='whitegrid')

# Selecionando as 5 principais fontes de recurso para o gráfico
fontes_principais = receitas_filtradas['Fonte Recurso'].value_counts().head(5).index
receitas_filtradas = receitas_filtradas[receitas_filtradas['Fonte Recurso'].isin(fontes_principais)]

# Filtrando apenas os anos com dados registrados
anos_disponiveis = receitas_filtradas['Data'].dt.year.unique()
anos_disponiveis = sorted(anos_disponiveis)  # Ordenando os anos

# Gráfico de linha para sazonalidade com datas completas
plt.figure(figsize=(14, 6))
sns.lineplot(data=receitas_filtradas, x='Data', y='Valor', hue='Fonte Recurso', linewidth=2.5)

# Ajuste de escala logarítmica no eixo y para melhor visualização
plt.yscale('log')

# Adicionando anotações e ajustando o eixo x
plt.title('Sazonalidade das Receitas ao Longo dos Anos (Top 5 Fontes)')
plt.xlabel('Data')
plt.ylabel('Valor (escala logarítmica)')
plt.xticks(pd.to_datetime([f'{ano}-01-01' for ano in anos_disponiveis]), rotation=45)

plt.tight_layout()  # Ajusta o layout para evitar sobreposição
plt.show()


