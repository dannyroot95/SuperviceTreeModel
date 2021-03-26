import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

df = pd.read_csv("datasets\supervivence.csv")

#

def accuracy(y_predicted, y_real):
    mask = np.array(y_predicted) == np.array(y_real)
    return mask.sum()/len(y_real)

#

    def predict_instance(x):
        prediction = 0

        if x.Sex == 'female':
        prediction = 1
    elif x.Pclass == 1:
        prediction = 1
    
    return prediction

#

def predict(X):
    y_predicted = []
    for x in X.itertuples(): 
        y_i = predict_instance(x) 
        y_predicted.append(y_i)
    return y_predicted

#

X = df.drop("Survived", axis=1)
y = df.Survived

#

y_pred = predict(X)

#

print("Accuracy final: ", round(accuracy(y_pred, y), 3))

# 
from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)

#

print(y.value_counts())
print(y.size)

#

no_sobrevivieron, sobrevivieron = y.value_counts()
N = y.size
gini_inicial = 1 - (no_sobrevivieron/N)**2 - (sobrevivieron/N)**2

#

print(gini_inicial)

#

mascara = df.Sex == 1
y_female = y[mascara]
y_male = y[~mascara]

#

muestras_neg, muestras_pos = y_female.value_counts()
N = y_female.size
gini_female = 1 - (muestras_neg/N)**2 - (muestras_pos/N)**2
print(gini_female)

#

muestras_neg, muestras_pos = y_male.value_counts()
N = y_male.size
gini_male = 1 - (muestras_neg/N)**2 - (muestras_pos/N)**2
print(gini_male)

#

print('Impureza Gini al separar por Genero:',(y_female.sum()*gini_female + y_male.sum()*gini_male)/y.size)

#

aux = np.zeros(2)
gini = np.zeros(3)
suma = 0


for i in range(0,3):
    for j in range(0,2):
        aux[j]=( (np.array(y[(df.Pclass==(i+1)) &   (df.Survived==j)]).size)/(np.array(y[(df.Pclass==(i+1))]).size))**2
    gini[i] = (1- aux[0] - aux [1])
    suma+=(y[df.Pclass==(i+1)].sum()*gini[i])
print('Impureza Gini al separar por clase:', suma/y.size)

#

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5)

#

clf.fit(X, y)

# Predecimos sobre nuestro set
for i in range(10):
    clf.fit(X, y)
    y_pred = clf.predict(X)

# Comaparamos con las etiquetas reales
    print('Accuracy:', accuracy_score(y_pred,y))

#

plt.figure(figsize = (10,8))
tree.plot_tree(clf, filled = True, feature_names= X.columns)
plt.show()

#

importances = clf.feature_importances_
columns = X.columns
sns.barplot(columns, importances)
plt.title('Importancia de cada Feature')
plt.grid()
plt.show()
