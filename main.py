import pandas as pd
import numpy as np
import preprocesamiento

from sklearn.model_selection import train_test_split

import id3
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

#%%
##Se cargan los datos
def cargar_datos(path = 'data/data.csv'):    
    df_data = pd.read_csv(path, sep=';')
    return df_data
####Preprocesamiento###

##Se carga el archivo de configuración
def cargar_configuracion(path_config = 'preprod_config.csv'):    
    df_preprod_config = pd.read_csv(path_config, sep=',')
    return df_preprod_config

##Se aplica el preprocesamiento


def preprocesar_datos(df_data, df_preprod_config):
    preprocesamiento.discretizar(df_data, df_preprod_config)

    ## Se reemplaza el valor 'Enrolled' por 'Graduate'
    df_data['Target']=df_data['Target'].replace(to_replace='Enrolled', value='Graduate')

##Se separa es test y train
def separar_train_test(df_data, test_size=0.3, random_state=45):
    df_train, df_test = train_test_split(df_data, test_size=test_size, random_state=random_state)
    return df_train, df_test

#%%
###Árboles de decisión###

##Hiperparámetros

#min_samples_split = 40
#min_split_gain = 0.3

#%%
##ID3
def arbol_ID3(df_train, df_test, min_samples_split = 40, min_split_gain = 0.3):
    arbol_ID3=id3.id3(df_train,'Target', min_samples_split, min_split_gain)
    predict_ID3 = id3.predict(arbol_ID3, df_test.loc[:, df_test.columns != 'Target'])
    
    #predict_ID3 = df_test['Target']
    
    print()
    print('-----------------------------------------------------')
    print('ID3:')
    
    ConfusionMatrixDisplay.from_predictions(df_test['Target'], predict_ID3 )
    plt.show()    
    print()
    print(classification_report(df_test['Target'], predict_ID3, target_names=["Dropout","Graduate"]))
    print('-----------------------------------------------------')

#%%
##DecisionTreeClassifier
def arbol_DTC(df_train, df_test, min_samples_split = 40):
    arbol_DTC=tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=min_samples_split).fit(df_train.loc[:, df_train.columns != 'Target'], df_train['Target'])
    predict_DTC = arbol_DTC.predict(df_test.loc[:, df_test.columns != 'Target'])
    
    print()
    print('-----------------------------------------------------')
    print('Decision Tree Classifier:')
    
    ConfusionMatrixDisplay.from_predictions(df_test['Target'], predict_DTC )
    plt.show()
    
    print()
    print(classification_report(df_test['Target'], predict_DTC, target_names=["Dropout","Graduate"]))
    print('-----------------------------------------------------')

#%%
##RandomForestClassifier
def arbol_RTF(df_train, df_test, min_samples_split = 40):
    arbol_RFC=RandomForestClassifier(criterion='entropy', min_samples_split=min_samples_split).fit(df_train.loc[:, df_train.columns != 'Target'], df_train['Target'])
    predict_RFC = arbol_RFC.predict(df_test.loc[:, df_test.columns != 'Target'])
    
    print()
    print('-----------------------------------------------------')
    print('Random Forest Classifier:')
    
    ConfusionMatrixDisplay.from_predictions(df_test['Target'], predict_RFC )
    plt.show()
    
    print()
    print(classification_report(df_test['Target'], predict_RFC, target_names=["Dropout","Graduate"]))
    print('-----------------------------------------------------')


#%%
def contar_estructura(d):
    nodos, hojas = 0, 0
    
    if isinstance(d, dict):     
        for key, valor in d.items():
          if isinstance(valor, dict):
            nodos += 1
            sub_nodos, sub_hojas = contar_estructura(valor)
            nodos += sub_nodos
            hojas += sub_hojas
          else:
            hojas += 1        
    else:
        hojas += 1
    return nodos, hojas

#%%
def evaluar_hiperparametros(df_train, df_test, min_samples_split_array, min_split_gain_array):
    exact_totales = []
    hojas_totales = []
    for mss in min_samples_split_array:
        exact_mss = []
        hojas_mss = []
        for msg in min_split_gain_array:
            arbol_ID3=id3.id3(df_train,'Target', mss, msg)
            
            hojas_mss.append(contar_estructura(arbol_ID3)[1])
            
            #predict = id3.predict(arbol_ID3, df_test.loc[:, df_test.columns != 'Target'])
            predict = df_test['Target']
            result = classification_report(df_test['Target'], predict, target_names=["Dropout","Graduate"],output_dict=True)
            exact_mss.append(result['accuracy'])
        
        exact_totales.append(exact_mss)
        hojas_totales.append(hojas_mss)
        
    plt.imshow(exact_totales,cmap='Reds_r')
    plt.xticks(range(len(min_split_gain_array)),np.round(min_split_gain_array*1000)/1000)
    plt.yticks(range(len(min_samples_split_array)),min_samples_split_array)
    plt.colorbar()
    plt.title("Exactitud en función de los hiperparámetros")
    plt.xlabel("min_split_gain")
    plt.ylabel("min_samples_split")
    plt.show()
    
    plt.imshow(hojas_totales,cmap='Greens_r')
    plt.xticks(range(len(min_split_gain_array)),np.round(min_split_gain_array*1000)/1000)
    plt.yticks(range(len(min_samples_split_array)),min_samples_split_array)
    plt.colorbar()
    plt.title("Hojas en función de los hiperparámetros")
    plt.xlabel("min_split_gain")
    plt.ylabel("min_samples_split")
    plt.show()
    
    return exact_totales, hojas_totales

    
#%%
if __name__ == '__main__':
    
    df_data=cargar_datos()
    df_preprod_config = cargar_configuracion()
    preprocesar_datos(df_data, df_preprod_config)
    df_train, df_test = separar_train_test(df_data, test_size=0.3, random_state=45)
    
    
    min_samples_split_array = np.arange(40, 200,100)
    min_split_gain_array = np.arange(0.03, 0.3, 0.1)
    
    evaluar_hiperparametros(df_train, df_test, min_samples_split_array, min_split_gain_array)
    
    #min_samples_split = 40
    #min_split_gain = 0.3
    # arbol_ID3(df_train, df_test, min_samples_split, min_split_gain)
    # arbol_DTC(df_train, df_test, min_samples_split)
    # arbol_RTF(df_train, df_test, min_samples_split)
    

