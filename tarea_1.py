# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 13:18:18 2023

@author: jrf_7
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

path = 'datos/data.csv'

df_data = pd.read_csv(path, sep=';')

columnas = df_data.columns

count = np.zeros(df_data.columns.shape)

for idx, col in enumerate(df_data.columns):
    count[idx]=df_data[col].nunique()

count = np.expand_dims(count, 0)    
df_count = pd.DataFrame(data=count, columns = df_data.columns)

df_count_grandes = df_count.copy(deep=True)
grater_columns=[]

for col in df_count:
    if df_count[col].sum()>90:
        grater_columns.append(col)
        
#%%
columnas_mas_datos=df_count.sum(axis=0)>90


#%%        
        

###Discretizar datos
df_data_discreta_propia=df_data.copy(deep=True)
num_categorias=10
nombres_cat = []

for i in range(num_categorias-1):
    nombres_cat.append(i)

for col in grater_columns:
    bins = np.linspace(df_data[col].min(),df_data[col].max(),num=num_categorias )
    nueva_col=pd.cut(df_data[col],bins, labels=nombres_cat, include_lowest=True)
    nueva_col = np.array(nueva_col)
    df_data_discreta_propia[col]=nueva_col


    
#%%    
###SKlearn discretization
df_data_discreta_SKlearn=df_data.copy(deep=True)
for col in grater_columns:
    X=np.array(df_data[col]).reshape(-1,1)
    enc = KBinsDiscretizer(n_bins=num_categorias, encode="ordinal").fit(X)
    df_data_discreta_SKlearn[col] = enc.transform(X)

#%%

df_data_discreta_SKlearn['Target']=df_data_discreta_SKlearn['Target'].replace(to_replace='Enrolled', value='Graduate')


#%%

positivos = df_data_discreta_SKlearn

def entropia(df:pd.DataFrame,col_origen, col_objetivo):
    df.groupby(by=[col_origen])
    
    
    
#%%

#ccc=df_data_discreta_SKlearn[['Marital status', 'Target']].groupby(by=['Marital status'])
ccc=df_data_discreta_SKlearn[['Marital status', 'Target']].groupby(by=['Marital status'])
total_graduados=ccc['Target'].value_counts()
total_graduados_cuenta=total_graduados/ccc.count()

#%%
hhh=(ccc['Target']=='Graduate').sum()/(ccc['Target']).sum()

#%%
g_sum = df_data_discreta_SKlearn.groupby('Marital status')['Target'].transform('count')

#%%


# Agrupamos por 'Grupo' y contamos la cantidad de veces que aparece el valor 1
counts = df_data_discreta_SKlearn.groupby('Marital status')['Target'].apply(lambda x: (x == 'Graduate').sum())

# Obtenemos el tamaño de cada grupo
sizes = df_data_discreta_SKlearn.groupby('Marital status').size()

# Calculamos el porcentaje
percentages = counts / sizes * 100

print(percentages)

counts_S=df_data_discreta_SKlearn.groupby('Target')['Target'].count()/df_data_discreta_SKlearn['Target'].count()
entropia_S=-counts_S['Dropout']*np.log2(counts_S['Dropout'])-counts_S['Graduate']*np.log2(counts_S['Graduate'])

#%%
counts_total = df_data_discreta_SKlearn['Target'].count()
counts_valor=df_data_discreta_SKlearn.groupby('Marital status')['Target'].count()
counts_pos = df_data_discreta_SKlearn.groupby('Marital status')['Target'].apply(lambda x: (x == 'Graduate').sum()/x.count())
counts_neg = df_data_discreta_SKlearn.groupby('Marital status')['Target'].apply(lambda x: (x == 'Dropout').sum()/x.count())
entropia_Marital = -counts_pos*np.log2(counts_pos)-counts_neg*np.log2(counts_neg)
ganancia_Marital = entropia_S- ((counts_valor/counts_total)*entropia_Marital).sum()

#%%

pct = lambda x:  x / x.sum()
out=df_data_discreta_SKlearn.groupby(['Marital status','Target']).sum().groupby('Target').apply(pct)

#%%
def ganancia(df:pd.DataFrame,col_origen,col_objetivo, entropia_S):
    counts_total = df[col_objetivo].count()
    counts_valor=df.groupby(col_origen)[col_objetivo].count()
    counts_pos = df.groupby(col_origen)[col_objetivo].apply(lambda x: (x == 'Graduate').sum()/x.count())
    counts_neg = df.groupby(col_origen)[col_objetivo].apply(lambda x: (x == 'Dropout').sum()/x.count())
    entropia_valor = -counts_pos*np.log2(counts_pos)-counts_neg*np.log2(counts_neg)
    ganancia = entropia_S - ((counts_valor/counts_total)*entropia_valor).sum()
    return ganancia
    
ganancia_Marital2 = ganancia(df_data_discreta_SKlearn[['Marital status','Target']], 'Marital status', 'Target', entropia_S )


#%%
col1=['Alta', 'Baja', 'Media', 'Media']
col2=['Graduate', 'Dropout', 'Graduate', 'Dropout']
col1=np.expand_dims(col1, 0).T
col2=np.expand_dims(col2, 0).T
datos= np.concatenate((col1, col2))
df_prueba = pd.DataFrame(data=[['Alta', 'Baja', 'Media', 'Media'],['Graduate', 'Dropout', 'Graduate', 'Dropout']], columns=['Marital Status', 'Target'])
