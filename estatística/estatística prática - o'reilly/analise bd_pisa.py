# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:41:32 2024

@author: Amilcar
"""

#analise bd_pisa

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
bd_pisa = pd.read_csv('bd_pisa.csv')

# Informações básicas
bd_pisa.info()

# Tratando os dados (Wrangling)
#%%
import pandas as pd

def col_met(df):
    """
    Identifica colunas tipificadas como strings ou categorias que possuem dados potencialmente métricos.
    
    Args:
        df (pd.DataFrame): DataFrame para análise.

    Returns:
        list: Lista com os nomes das colunas que podem conter dados numéricos, mas não estão tipificadas como numéricas.
    """
    colunas_metricas_potenciais = []

    for coluna in df.columns:
        # Verifica se a coluna não é numérica
        if not pd.api.types.is_numeric_dtype(df[coluna]):
            try:
                # Tenta converter todos os valores da coluna para números
                convertidos = pd.to_numeric(df[coluna], errors='coerce')
                
                # Verifica se pelo menos um valor foi convertido para numérico
                if convertidos.notna().any():
                    colunas_metricas_potenciais.append(coluna)
            except ValueError:
                # Ignora colunas que não podem ser convertidas
                pass
    
    return colunas_metricas_potenciais

#%%
colunas_metricas_bd_pisa = col_met(bd_pisa)
print(colunas_metricas_bd_pisa)

# Criando uma cópia do DataFrame com as colunas convertidas
bd_pisa_tipok = bd_pisa.copy()

# Criar uma cópia e aplicar a conversão diretamente nas colunas
bd_pisa_tipok[colunas_metricas_bd_pisa] = bd_pisa[colunas_metricas_bd_pisa].apply(pd.to_numeric, errors='coerce')

# Info da nova lista com os tipos dos dados corrigidos
bd_pisa_tipok.info()

#salvando o arquivo
bd_pisa_tipok.to_csv('bd_pisa_tipok.csv', index = False)
#%%

pisa_metric = bd_pisa_tipok.select_dtypes(include=['number']).columns.tolist()
pisa_catego = bd_pisa.tipok.select_dtypes(include=['object']).columns.tolist()

# Sumarizando as variáveis numéricas
anacorr_pisa = bd_pisa_tipok[pisa_metric].corr()
anadesc_pisa = bd_pisa_tipok[pisa_metric].describe().T

#%%
import numpy as np

# Máscara booleana para a parte superior da matriz de correlação (exclui a diagonal)
mask = np.triu(np.ones(anacorr_pisa.shape), k=1).astype(bool)

# Aplicar a máscara para considerar apenas a parte superior da matriz
pares_correlacionados_pisa = (
    anacorr_pisa.where(mask)  # Filtra com a máscara booleana
    .stack()  # Converte a matriz de correlação em uma série
    .reset_index()  # Converte para DataFrame
)
pares_correlacionados_pisa.columns = ['Variável 1', 'Variável 2', 'Correlação']

# Filtrar os pares que atendem ao critério
pares_filtrados_pisa = pares_correlacionados_pisa[
    (pares_correlacionados_pisa['Correlação'] > 0.85) | (pares_correlacionados_pisa['Correlação'] < -0.85)
]

print(pares_filtrados_pisa)


#%%
# Filtrar apenas as variáveis envolvidas nos pares
variaveis_interessantes_pisa = list(set(pares_filtrados_pisa['Variável 1']).union(pares_filtrados_pisa['Variável 2']))
sns.pairplot(bd_pisa_tipok[variaveis_interessantes_pisa])
plt.show()


