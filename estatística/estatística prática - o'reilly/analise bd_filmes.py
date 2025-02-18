# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:41:32 2024

@author: Amilcar
"""

#analise bd_filmes

pip install pandas numpy matplotlib seaborn xlrd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio_renderes_defaultl = "browser" 

# Lendo o banco de dados
bd_filmes = pd.read_csv('bd_filmes.csv')

# Informações básicas
bd_filmes.info()

# Tratando os dados (Wrangling)
#%%

filmes_metric = bd_filmes.select_dtypes(include=['number']).columns.tolist()
filmes_catego = bd_filmes.select_dtypes(include=['object']).columns.tolist()

# Sumarizando as variáveis numéricas
anacorr_filmes = bd_filmes[filmes_metric].corr()
anadesc_filmes = bd_filmes[filmes_metric].describe().T

# Sumarizando as variáveis categóricas
anadesccat_filmes = bd_filmes[filmes_catego].describe().T

#%% Contagem das categorias
for coluna in filmes_catego:
    print(f"Distribuição de categorias na variável {coluna}:")
    print(filmes_catego[coluna].value_counts())
    print("\n")
    
#%% Crosstable - Tabela de contingência
# Tabela cruzada para duas variáveis categóricas
anacros_bd_filmes_genresxage = pd.crosstab(bd_filmes['Genres'], bd_filmes['Age'])
anacros_bd_filmes_genresxcountry = pd.crosstab(bd_filmes['Genres'], bd_filmes['Country'])

#%% plotando a contagem das variáveis em um gráfico de barras
for coluna in filmes_catego.columns:
    filmes_categ[coluna].value_counts().plot(kind='bar', figsize=(8, 5), title=f"Distribuição de {coluna}")
    plt.xlabel("Categorias")
    plt.ylabel("Frequência")
    plt.xticks(rotation=45)
    plt.show()
#%%
import numpy as np

# Máscara booleana para a parte superior da matriz de correlação (exclui a diagonal)
mask = np.triu(np.ones(anacorr_filmes.shape), k=1).astype(bool)

# Aplicar a máscara para considerar apenas a parte superior da matriz
pares_correlacionados_filmes = (
    anacorr_filmes.where(mask)  # Filtra com a máscara booleana
    .stack()  # Converte a matriz de correlação em uma série
    .reset_index()  # Converte para DataFrame
)
pares_correlacionados_filmes.columns = ['Variável 1', 'Variável 2', 'Correlação']

# Filtrar os pares que atendem ao critério
pares_filtrados_filmes = pares_correlacionados_filmes[
    (pares_correlacionados_filmes['Correlação'] > 0.85) | (pares_correlacionados_filmes['Correlação'] < -0.85)
]

print(pares_filtrados_filmes)


#%%
# Filtrar apenas as variáveis envolvidas nos pares
variaveis_interessantes_filmes = list(set(pares_filtrados_filmes['Variável 1']).union(pares_filtrados_filmes['Variável 2']))
sns.pairplot(bd_filmes[variaveis_interessantes_filmes])
plt.show()


