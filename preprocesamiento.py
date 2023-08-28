from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np

def discretizar(df_data:pd.DataFrame, df_preprod_config:pd.DataFrame):
    
    print()
    print('-----------------------------------------------------------------------')
    print('--------------------------Bordes de los bines--------------------------')
    
    for idx, col in enumerate(df_preprod_config['columna']):
        if df_preprod_config['algoritmo'][idx]=='redondeo':
            df_data[col]= np.round(df_data[col])
            
        if df_preprod_config['algoritmo'][idx]=='bins':
            X=np.array(df_data[col]).reshape(-1,1)
            enc = KBinsDiscretizer(n_bins=df_preprod_config['bins'][idx], encode="ordinal").fit(X)        
            df_data[col] = enc.transform(X)
            print()
            print(f'Columna: {col}')
            print(enc.bin_edges_[0])
    
    print()
    print('-----------------------------------------------------------------------')
    print()


#Para pruebas

# path_config = 'preprod_config.csv'
# df_preprod_config = pd.read_csv(path_config, sep=',')
# path = 'data/data.csv'
# df_data = pd.read_csv(path, sep=';')     
# contar_unicos_antes= df_data.nunique()
# discretizar(df_data, df_preprod_config)
# contar_unicos_despues= df_data.nunique()