# Modelo predictivo de arbol de descicion si una persona va a sobrevivir o no
# Importamos las librerias y la base de datos 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

df = pd.read_csv("datasets\supervivence.csv")

# La función `accuracy`, dada las etiquetas que ustedes predigan y las etiquetas reales, calcula la medida de performance, en este caso, la exactitud. 
#**No la tienes que modificar a su implementación**.

def accuracy(y_predicted, y_real):
    mask = np.array(y_predicted) == np.array(y_real)
    return mask.sum()/len(y_real)

#La función predict_instance, dada una instancia x con sus atributos, predice si sobrevivió o no. Es la única función que tendrás que modificar.

    def predict_instance(x):


# Modificar las siguientes líneas de codigo. 
# Este será su algoritmo algoritmo para predecir si sobrevivirá o no por instancia.
# La variable prediction debe contener la etiqueta 0 o 1 
# Algunas opciones son: predecir que nadie sobrevivio, que todos sobrevivieron,
# predecir al azar, y usar lo aprendido cuando exploramos el dataset 

        prediction = 0 # cambiar

            ### UNA POSIBLE FORMA DE EMPEZAR:
#     if x.Age < 12:
#         prediction = 1
#     else:
#         prediction = 0
#     # FIN DE COMPLETAR
    
    ### Si usamos el genero y la clase

        if x.Sex == 'female':
        prediction = 1
    elif x.Pclass == 1:
        prediction = 1
    
    return prediction

    # Por último, la función predict toma todo las instancias X y, usando la función que definieron antes, 
    # predice para cada una de ellas si sobrevivió o no. No la tienes que modificar.


def predict(X):
    y_predicted = []
    for x in X.itertuples(): 
        y_i = predict_instance(x) 
        y_predicted.append(y_i)
    return y_predicted

# Cargamos el dataset  y separarmos en una variable X los atributos que usarás para predecir,
# y en una variable y la etiqueta que quieres predecir. En este caso, si sobrevivió o no.

X = df.drop("Survived", axis=1)
y = df.Survived

# Usar los datos X para predecir si las personas sobrevivieron o no utilizando la función predict.
# No tienes que modificar ninguna de las funciones por ahora.

y_pred = predict(X)

# Calcula la medida de performance entre las etiquetas reales y y las etiquetas predichas y_pred con la función accuracy.

print("Accuracy final: ", round(accuracy(y_pred, y), 3))

# Calcula la matriz de confusión con Scikit-Learn. 

from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)

# Ahora se va a calcular cuán buena es la pregunta del género y clase para separar las muestras usando la impureza Gini. Para ello:
# se calcula la impureza inicial del dataset. en la variable y la que separaste las etiquetas. 

print(y.value_counts())
print(y.size)


no_sobrevivieron, sobrevivieron = y.value_counts()
N = y.size
gini_inicial = 1 - (no_sobrevivieron/N)**2 - (sobrevivieron/N)**2

#calcula la impureza Gini luego de separar por el género. se tienen que calcular la impureza en dos hojas - una correspondiente a género masculino y otra al femenino
# - y luego hacer un promedio ponderado. 

print(gini_inicial)

mascara = df.Sex == 1
y_female = y[mascara]
y_male = y[~mascara]

muestras_neg, muestras_pos = y_female.value_counts()
N = y_female.size
gini_female = 1 - (muestras_neg/N)**2 - (muestras_pos/N)**2
print(gini_female)

muestras_neg, muestras_pos = y_male.value_counts()
N = y_male.size
gini_male = 1 - (muestras_neg/N)**2 - (muestras_pos/N)**2
print(gini_male)


print('Impureza Gini al separar por Genero:',(y_female.sum()*gini_female + y_male.sum()*gini_male)/y.size)

aux = np.zeros(2)
gini = np.zeros(3)
suma = 0


for i in range(0,3):
    for j in range(0,2):
        aux[j]=( (np.array(y[(df.Pclass==(i+1)) &   (df.Survived==j)]).size)/(np.array(y[(df.Pclass==(i+1))]).size))**2
    gini[i] = (1- aux[0] - aux [1])
    suma+=(y[df.Pclass==(i+1)].sum()*gini[i])
print('Impureza Gini al separar por clase:', suma/y.size)


# Creamos un objeto arbol

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5)


clf.fit(X, y)

# Predecimos sobre nuestro set
for i in range(10):
    clf.fit(X, y)
    y_pred = clf.predict(X)

# Comaparamos con las etiquetas reales
    print('Accuracy:', accuracy_score(y_pred,y))


plt.figure(figsize = (10,8))
tree.plot_tree(clf, filled = True, feature_names= X.columns)
plt.show()


importances = clf.feature_importances_
columns = X.columns
sns.barplot(columns, importances)
plt.title('Importancia de cada Feature')
plt.grid()
plt.show()