# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:23:53 2024

@author: Amilcar
"""

"""
Tipos de dados
    Dados contínuos (intervalo, flutuação, numérico) são dados que podem assumir qualquer valor em um intervalo
    Dados discretos (inteiros, contagem) poem apenas assumir valores inteiros, como contagens.
    Categóricos (enumeração, fatores, nominal, politômico) podem assumir apenas um conjunto específico de valores representando um conjunto de categorias
        Binários tem apenas duas categorias de valores: 1/0 ou verdadeiro/falso.
        Ordinais (fator ordenado) tem uma ordem explícita.

Dados retangulares (tabelas e planilhas)
    Variáveis são representadas usualmente nas colunas (características, atributos, entradas, indicadores)
    Observações são representadas usualmente nas linhas (registros, casos, exemplos, padrões, amostras)

    Os dados categóricos costumam ser resumidos em proporções, e podem ser visualizados em um gráfico de barras.
    O valor esperado é a soma dos valores vezes sua probabilidade de ocorrência, e costuma ser usado para resumir os níveis de uma variável de fator.
    O coeficiente de correlação é uma métrica que mede o nível em que as variáveis númericas estão associadas uma às outras entre -1 e +1.
    Uma matriz de correlação é uma planilha com variáveis tanto nas linhas, quanto nas colunas, e os valores das células correspondem às correlações entre as variáveis.
    O diagrama de dispersão é um gráfico tendo os valores de duas variáveis como as coordenadas (x, y) dos pontos.
    Quando existe relação direta entre os valores das duas variáveis, diz-se que a correlação é positiva, se não, é negativa.
    O coeficiente de correlação é uma métrica padronizada, então varia de -1 a 1.

Medidas de posição (localização)
    A métrica básica para localização é a média, mas esta pode ser sensível a valores extremos (outlier).
    
Métricas de variabilidade
    Desvios (erros, resíduos) são as diferenças entre os valores observados e a estimativa de localização.

Explorando a distribuição
    Histograma é um gráfico de frequências com as colunas (variáveis) em x e a contagem (frequência de ocorrências) em y.
    O boxplot é um gráfico, elaborado por Tukey, para visualizar a distribuição de dados.
    A tabela de frequência é um registro de contagem de valores que caem em um conjunto de intervalos (colunas).
    Gráfico de densidade é uma versão simplificada do histograma, frequentemente usada em estimativas da densidade Kernel.
    
Explorando duas ou mais variáveis categóricas
    As tabelas de contigência (crosstable) são a ferramenta-padrão para a observação de contagens de duas variáveis categóricas.
"""
#%%
!pip install numpy 
!pip install pandas 
!pip install matplotlib 
!pip install seaborn
!pip install xlrd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio_renderes_defaultl = "browser" 

#%%
bd_pisa = pd.read_csv('bd_pisa.csv')
#bd_bank = pd.read.xls('bd_bank.xls')
#bd_pjmc = pd.read.xls('bd_pjmc.xls')
bd_distritos = pd.read_excel('bd_distritos.xlsx')

#%%
bd_pisa.info()
bd_distritos.info()
bd_pisa.columns
bd_distritos.columns

#%% Resumindo os dados

## 1. Visualizando estatísticas descritivas

# Tabela de descritivas para variáveis quantitativas

dados_tempo.describe()

# Estatísticas individuais

dados_tempo['tempo'].count() # contagem
dados_tempo['tempo'].mean() # média
dados_tempo['tempo'].median() # mediana
dados_tempo['tempo'].min() # mínimo
dados_tempo['tempo'].max() # máximo
dados_tempo['tempo'].std() # desvio padrão
dados_tempo['tempo'].var() # variância
dados_tempo['tempo'].quantile([0.25, 0.75]) # quartis
dados_tempo['tempo'].sum() # soma

# Matriz de correlações de Pearson

dados_tempo[['tempo', 'distancia', 'semaforos']].corr()

# Tabela de frequências para variáveis qualitativas

dados_tempo['periodo'].value_counts() # frequências absolutas
dados_tempo['perfil'].value_counts(normalize=True) # frequências relativas

# Tabela de frequências cruzadas para pares de variáveis qualitativas

pd.crosstab(dados_tempo['periodo'], dados_tempo['perfil'])
pd.crosstab(dados_tempo['periodo'], dados_tempo['perfil'], normalize=True)

## 2. Obtendo informações de valores únicos das variáveis

dados_tempo['tempo'].unique()
dados_tempo['periodo'].unique()
dados_tempo['perfil'].nunique() # quantidade de valores únicos

## 3. Criando um banco de dados agrupado (um critério)

dados_periodo = dados_tempo.groupby(['periodo'])

# Gerando estatísticas descritivas

dados_periodo.describe()

# Caso a tabela gerada esteja com visualização ruim no print, pode transpor

dados_periodo.describe().T

# Tamanho de cada grupo

dados_periodo.size()

# Criando um banco de dados agrupado (mais de um critério)

dados_criterios = dados_tempo.groupby(['periodo', 'perfil'])

# Gerando as estatísticas descritivas

dados_criterios.describe().T

# Tamanho de cada grupo

dados_criterios.size()

# Especificando estatísticas de interesse

dados_periodo.agg({'tempo': 'mean',
                   'distancia': 'mean',
                   'periodo': 'count'})
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

# Exemplo de uso
df = pd.DataFrame({
    'idade': ['25', '30', '35', '40'],     # String, mas numérica
    'nome': ['João', 'Maria', 'José', 'Ana'],  # Não numérica
    'salario': ['1500.50', '2000.75', '3000', '2500'],  # String, mas numérica
    'categoria': ['A', 'B', 'C', 'D'],    # Não numérica
    'pontos': [10, 20, 30, 40]            # Já numérica
})

colunas_metricas_potenciais = col_met(df)
print("Colunas com dados métricos potenciais:", colunas_metricas_potenciais)

#%%
colunas_metricas_bd_pisa = col_met(bd_pisa)

# Criando uma cópia do DataFrame com as colunas convertidas
bd_pisa_tipok = bd_pisa.copy()

# Criar uma cópia e aplicar a conversão diretamente nas colunas
bd_pisa_tipok = bd_pisa.copy()
bd_pisa_tipok[colunas_metricas_bd_pisa] = bd_pisa[colunas_metricas_bd_pisa].apply(pd.to_numeric, errors='coerce')
#%%
#Resumindo os dados numericos
colunas_metricas_bdpisatipok = bd_pisa_tipok.select_dtypes(include=['number']).columns.tolist()
print("Colunas métricas:", colunas_metricas_bdpisatipok)

bd_pisa_anacorr = bd_pisa_tipok[colunas_metricas_bdpisatipok].corr()
bd_pisa_anadesc = bd_pisa_tipok[colunas_metricas_bdpisatipok].describe()

#%%

import matplotlib.pyplot as pltp

# Medidas descritivas
desc_stats = bd_pisa_anadesc.copy().T  # Transpor para melhor organização

# Selecionando as medidas de interesse
medidas = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']

# Verificar se as colunas existem
print("Colunas disponíveis:", desc_stats.columns.tolist())

# Criar o gráfico de barras agrupadas
desc_stats[medidas].plot(kind='bar', figsize=(12, 6))
pltp.title("Comparação das Estatísticas Descritivas")
pltp.ylabel("Valores")
pltp.xlabel("Variáveis")
pltp.xticks(rotation=45)  # Inclina os rótulos no eixo X
pltp.legend(title="Medidas")
pltp.tight_layout()
pltp.show()


 





