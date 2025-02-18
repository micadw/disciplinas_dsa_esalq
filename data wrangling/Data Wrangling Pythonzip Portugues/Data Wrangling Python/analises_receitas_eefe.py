# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:52:03 2024

@author: Amilcar
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

receitas = pd.concat([
    pd.read_csv('receitas2010.csv', encoding='latin1', sep=';', on_bad_lines='skip'),
    pd.read_csv('receitas2014.csv', encoding='latin1', sep=';', on_bad_lines='skip'),
    pd.read_csv('receitas2018.csv', encoding='latin1', sep=';', on_bad_lines='skip'),
    pd.read_csv('receitas2022.csv', encoding='latin1', sep=';', on_bad_lines='skip')
]).reset_index(drop=True)

receitas_fontes = receitas.groupby('Fonte Recurso')

receitas_ano_sel = receitas.query('Ano in [2010,2014,2018,2022]')
receitas_ano_sel.drop(columns=['valor'], inplace=True)

receitas_fontes['Fonte Recurso'].value_counts()
receitas_fontes['Fonte Recurso'].value_counts(normalize=True)
receitas['Ano'] = receitas['Ano'].astype('category')

# Converter valores em reais para float
receitas_ano_sel['Valor'] = (
    receitas_ano_sel['Valor']
    .str.replace('.', '', regex=True)  # Remove separadores de milhar
    .str.replace(',', '.', regex=True)  # Substitui vírgula por ponto
    
)


tabela_cruzada = pd.crosstab(receitas_ano_sel['Ano'], receitas_ano_sel['Fonte Recurso'])

tabela_soma = pd.crosstab(
    receitas_ano_sel['Ano'],              # Índice (linhas)
    receitas_ano_sel['Fonte Recurso'],    # Colunas
    values=receitas_ano_sel['Valor'],     # Valores a agregar
    aggfunc='sum'                 # Função de agregação
).fillna(0)                       # Preencher valores NaN com 0


receitas.columns
receitas_ano_sel.info()

receitas_describe = df_receitaseefe['Fonte Recurso'].value_counts()

receitas_fonte = receitas.groupby(by=['Fonte Recurso'])


# Configuração do estilo dos gráficos
sns.set(style='whitegrid')

# Convertendo a coluna 'Data' para datetime e extraindo o ano
receitas['Data'] = pd.to_datetime(receitas['Data'])
receitas['Ano'] = receitas['Data'].dt.year

# Convertendo a coluna 'Valor' de string para float
receitas_ano_sel['Valor'] = receitas_ano_sel['Valor'].str.replace(',', '.').astype(float)

# Criando faixas de valores
bins = [0, 1000, 5000, 10000, 50000, 100000, receitas['Valor'].max()]
labels = ['0-1K', '1K-5K', '5K-10K', '10K-50K', '50K-100K', '100K+']
receitas['Faixa de Valor'] = pd.cut(receitas['Valor'], bins=bins, labels=labels)

# Filtrando fontes de recurso com valores totais significativos
valor_por_fonte = receitas.groupby('Fonte Recurso')['Valor'].sum()
fontes_significativas = valor_por_fonte[valor_por_fonte > valor_por_fonte.quantile(0.1)].index
receitas_filtradas = receitas[receitas['Fonte Recurso'].isin(fontes_significativas)]

# Filtrando apenas os anos de interesse
anos_interesse = [2010]
receitas_filtradas = receitas_filtradas[receitas_filtradas['Ano'].isin(anos_interesse)]

# Gráfico de linha para sazonalidade com faixas de valores
plt.figure(figsize=(14, 6))
sns.lineplot(data=receitas_filtradas, x='Ano', y='Valor', hue='Fonte Recurso')
plt.title('Sazonalidade das Receitas ao Longo dos Anos')
plt.xlabel('Ano')
plt.ylabel('Valor (por faixas)')
plt.xticks(anos_interesse)  # Ajustando o eixo x para mostrar apenas os anos desejados
plt.show()


