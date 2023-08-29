
from collections import defaultdict
import numpy as np
import pandas as pd
##Funcion recursiva




def decision_tree(df,target_column, min_samples_split, min_split_gain):

    ## creo diccionario,llamo a id3, guardo resultado
    tree=id3(df,target_column, min_samples_split, min_split_gain)
    return tree


def id3(df,target_column, min_samples_split, min_split_gain):

    ## si no me quedan atributos, poner el valor mas comun
    root={}
    len_columns = len(df.columns)
    most_common_value = df[target_column].mode().values[0]
    # if most_common_value is None:
    #     if df[target_column].nunique()==1:
            #most_common_value= df[target_column].unique()


    #print("Valor mas comun: ", most_common_value)
    #print("Numero de columnas: ", len_columns)
    if len_columns == 1:
        return most_common_value
    
    #print("Numero de columnas: ", len_columns)
    #print("columnas: ", df.columns)

    ## si todos los ejemplos tienen el mismo valor de target, poner ese valor
    #print("Numero de valores unicos: ", df[target_column].nunique())
    if df[target_column].nunique() == 1:
        return most_common_value
    
    ## quedarme con el mejor atributo


    gain, best_attribute = get_best_attribute(df,target_column)
    #print(gain, best_attribute)

    ## check min samples split y  min split gain y si no se cumplen, poner el mas comun

    if gain < min_split_gain:
        return most_common_value
    if len(df) < min_samples_split:
        return most_common_value
    

    attribute_values=df[best_attribute].unique()
    #print("Atributo: ", best_attribute,attribute_values)
    
    root[best_attribute]={}

    for attribute in attribute_values:


        new_df = df[df[best_attribute]==attribute]
        new_df = new_df.drop(best_attribute, axis=1)
        if len(new_df) == 0:
            root[best_attribute][attribute] = most_common_value
        root[best_attribute][attribute] = id3(new_df,target_column, min_samples_split, min_split_gain)
    
    return root
    
    

    ## si se cumplen, generar una rama por cada valor posible del atributo
    ## filtrarel dataset por el valor de la rama y llamar recursivamente a la funcion
    ## si es vacio pongo el valor mas probable para el atributo



def get_best_attribute(df,target_column):
    max_gain = -1
    best_attribute = None
    for attribute in df.columns:
        if attribute != target_column:
            gain = entropy_gain(df,attribute,target_column)
            if gain > max_gain:
                max_gain = gain
                best_attribute = attribute
    return max_gain, best_attribute



def entropy(data):
    """Calculates the entropy of a dataset."""
    counts = data.value_counts(normalize=True)
    return -sum(counts * np.log2(counts))

def entropy_gain(data, attribute, target_column):
    """Calculates the entropy gain of an attribute in a dataset."""
    total_entropy = entropy(data[target_column])
    attribute_values = data[attribute].unique()
    weighted_entropy_sum = 0
    for value in attribute_values:
        subset = data[data[attribute] == value]
        weighted_entropy_sum += len(subset) / len(data) * entropy(subset[target_column])
    return total_entropy - weighted_entropy_sum



def test_decision_tree():
    """Tests the decision tree."""
    # Create a sample dataset
    df = pd.DataFrame({
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        'Play': [False, False, True, True, True, False, True, False, True, True, True, True, True, False]
    })

    # Train the decision tree
    tree = decision_tree(df, 'Play', min_samples_split=2, min_split_gain=0)

    print(tree)

def predict(tree, df):
    results=[]
    for index,row in df.iterrows():
        ##iterative predictio 
        root=list(tree.keys())[0]
        sub_tree=tree
        i=0
        while i in range(len(row)):


            valor = row[root]
            sub_tree=sub_tree[root][valor]

            if not isinstance(sub_tree, dict):
                results.append(sub_tree)
                break 
            root=list(sub_tree.keys())[0]
           
            
            i+=1
    return results       
                    
                



            

if __name__ == '__main__':
    test_decision_tree()

    df_test=   df = pd.DataFrame({
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        'Play': [False, False, True, True, True, False, True, False, True, True, True, True, True, False]

    })
    tree=decision_tree(df_test, 'Play', min_samples_split=2, min_split_gain=0)
    predict(tree, df_test)
