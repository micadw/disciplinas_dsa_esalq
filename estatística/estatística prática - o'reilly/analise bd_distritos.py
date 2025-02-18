# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:41:32 2024

@author: Amilcar
"""

#analise bd_distritos

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
bd_distritos = pd.read_excel('bd_distritos.xlsx')

# Informações básicas
bd_distritos.info()


# Tratando os dados (Wrangling)
distritos_metric = bd_distritos.select_dtypes(include=['number']).columns.tolist()
distritos_catego = bd_distritos.select_dtypes(include=['object']).columns.tolist()

""" Removendo colunas pseudometricas 
(números representando categorias ou que não podem ser 
 resumidos por medidas de posições como médias)""" 
distritos_metric.remove('cód_ibge')

# Sumarizando as variáveis numéricas
anacorr_distritos = bd_distritos[distritos_metric].corr()
anadesc_distritos = bd_distritos[distritos_metric].describe().T
anahist_distritos_renda = bd_distritos['renda'].plot.hist()

#%%
import numpy as np

# Máscara booleana para a parte superior da matriz de correlação (exclui a diagonal)
mask = np.triu(np.ones(anacorr_distritos.shape), k=1).astype(bool)

# Aplicar a máscara para considerar apenas a parte superior da matriz
pares_correlacionados = (
    anacorr_distritos.where(mask)  # Filtra com a máscara booleana
    .stack()  # Converte a matriz de correlação em uma série
    .reset_index()  # Converte para DataFrame
)
pares_correlacionados.columns = ['Variável 1', 'Variável 2', 'Correlação']

# Filtrar os pares que atendem ao critério
pares_filtrados = pares_correlacionados[
    (pares_correlacionados['Correlação'] > 0.85) | (pares_correlacionados['Correlação'] < -0.85)
]

print(pares_filtrados)


#%%
# Filtrar apenas as variáveis envolvidas nos pares
variaveis_interessantes = list(set(pares_filtrados['Variável 1']).union(pares_filtrados['Variável 2']))
sns.pairplot(bd_distritos[variaveis_interessantes])
plt.show()


