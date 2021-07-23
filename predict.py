import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

#-----------------------------Despues del entrenamiento------------------------------------------
longitud, altura = 150, 150 #La misma definida antes
modelo = './modelo/modelo.h5' #Se entrega la direccion del modelo ya entrenado
pesos_modelo = './modelo/pesos.h5' #Se entrega la direccion del modelo
cnn = load_model(modelo) # Se le carga el modelo que esta almacenado
cnn.load_weights(pesos_modelo) #Se le cargan los pesos del modelo

#-----------------------------Funcion de Prediccion---------------------------------------------

def predict(file):
  x = load_img(file, target_size=(longitud, altura))#Se carga la imagen a analizar
  x = img_to_array(x) #Se convierte en arreglo la imagen 
  x = np.expand_dims(x, axis=0) #En la primera dimension de anade una dimension extra
  array = cnn.predict(x) #Se quiere hacer una prediccion sobre la imagen x y traera un arreglo de 2 dimensiones y  1 en la posicion correcta ejm=[1,0,0]
  result = array[0] #Nos interesa la posicion 0 del arreglo donde se almacena el resultado
  answer = np.argmax(result) #Trae el indice del valor mas alto que se trae en resultado [1,0,0], answer devolvera 0 donde se encuentra el valor 1
  if answer == 0: 
    print("pred: Perro")
  elif answer == 1:
    print("pred: Gato")
  elif answer == 2:
    print("pred: Gorila")

  return answer

#predict() ingresar nombre de la imagen a predecir
