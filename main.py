import pandas as pd
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
    #predict_ID3 = arbol_ID3.predict(df_test.loc[:, df_test.columns != 'Target'])
    
    print()
    print('-----------------------------------------------------')
    print('ID3:')
    
    #ConfusionMatrixDisplay.from_predictions(df_test['Target'], predict_ID3 )
    #plt.show()    
    print()
    # print(classification_report(df_test['Target'], predict_ID3, target_names=["Dropout","Graduate"]))
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
if __name__ == '__main__':
    
    df_data=cargar_datos()
    df_preprod_config = cargar_configuracion()
    preprocesar_datos(df_data, df_preprod_config)
    df_train, df_test = separar_train_test(df_data, test_size=0.3, random_state=45)
    min_samples_split = 40
    min_split_gain = 0.3
    
    arbol_ID3(df_train, df_test, min_samples_split, min_split_gain)
    arbol_DTC(df_train, df_test, min_samples_split)
    arbol_RTF(df_train, df_test, min_samples_split)
    
    
    