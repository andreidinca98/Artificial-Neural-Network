#IMPORTING DATASETS
import numpy as np 
import pandas as pd
import tensorflow as tf
import os
data1 = pd.read_csv(r"C:\..\..\..\Churn_Modelling.csv") 
x = data1.iloc [: , 3:-1].values #SEPARATE INDEPENDENT VARIABLES
#print(x)
y = data1.iloc [: , -1].values #SEPARATE DEPENDENT VARIABLES
#print(y)



#ENCODING CATEGORICAL DATA -- GENDER COLUMN
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[: , 2] = le.fit_transform( x[: , 2] )
x = np.asarray(x).astype('float32')


#ONE HOT ENCODING -- GEOGRAPHY COLUMN
from sklearn.compose import ColumnTransoformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransoformer(transformers=[('encoder',OneHotEncoder(),[1]]), remainder='passthrough')
X=np.array(ct.fit_transform(X))



#LET'S SPLIT INTO TRAIN SET & TEST SET
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 0 )

print(x_train)
#print("------")
print(x_test)
#print("------")
print(y_train)
#print("------")
print(y_test)


#APPLYING FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)
#must be updated-------------------------

#BULDING THE ANN
#INITIALZING THE ANN
ann = tf.keras.models.Sequential()

#ADDING THE INPUT LAYER &THE FIRST HIDDEN LAYER
ann.add(tf.keras.layers.Dense(units = 6 , activation = 'relu'))
#--units(the number of neurons--hyper-parameter--can be optimized)

#ADDING THE HIDDEN LAYERS
ann.add(tf.keras.layers.Dense(units = 6 , activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 6 , activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 6 , activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 6 , activation = 'relu'))
ann.add(tf.keras.layers.Dense(units = 6 , activation = 'relu'))





#ADDING THE OUPUT LAYER
ann.add(tf.keras.layers.Dense(units = 1),activation = 'sigmoid') 


#rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.0001 , rho = 0.9 , epsilon = 1e-07 , centered = False)


#TRAINING THE ANN
#1)compiling the ann
ann.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
#mean_squared_error is specific loss fuction for regresion 


#2)training the ann on the training set
ann.fit(x_train , y_train , batch_size = 32 , epochs = 100)   #default value for batch_size


#MAKING THE PREDICTIONS & EVALUATING THE MODEL
y_pred = ann.predict(x_test)
y_pred = (y_pred>0.5)
print (np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))



#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
