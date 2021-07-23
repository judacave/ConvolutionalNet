import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator #Ayuda a Preprocesar las imagenes
from tensorflow.python.keras import optimizers #Optimizador para entrenar el algoritmo
from tensorflow.python.keras.models import Sequential #Libreria para hacer redes neuronales secuenciales
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D #Capas de convolusion y Maxpooling
from tensorflow.python.keras import backend as K #Para limpiar sesiones de Keras y empezar de 0

K.clear_session() #Se limpia para empezar de 0


#Directorios donde estan ubicadas las imagenes de entrenamiento y validacion
data_entrenamiento = './data/entrenamiento'
data_validacion = './data/validacion'

"""
Parameters
"""
epocas=20 #Numero de veces que se va a iterar en el set de datos
longitud, altura = 150, 150 #Tamaño de Procesamiento de las imagenes
batch_size = 32 #Numero de imagenes que va a procesar en cada paso
pasos = 1000 #Numero de veces que se va a procesar la informacion en cada epoca
validation_steps = 300 #Se verifica que tan bien esta aprendiendo el algoritmo
filtrosConv1 = 32 #Numero de filtros en cada convolusion //PROFUNDIDAD DE 32
filtrosConv2 = 64 #PROFUNDIDAD DE 64
tamano_filtro1 = (3, 3) #Tamano de filtro o Kernel
tamano_filtro2 = (2, 2) #
tamano_pool = (2, 2) #Tamano del filtro en maxpooling
clases = 3 #Gato-Perro-Gorila
lr = 0.0004 #Learning rate, normalmente es un numero pequeno


#------------------------------------------------Preparamos nuestras imagenes-----------------------------------------------

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255, #Cada pixel tiene un rango de 0 a 1
    shear_range=0.2, #Generar imagenes inclinadas 
    zoom_range=0.2, #Les hara zoom a algunas para que aprenda que a veces apareceran solo sesiones del animal
    horizontal_flip=True) #Invertira algunas imagenes

test_datagen = ImageDataGenerator(rescale=1. / 255) #Se le entregan las imagenes tal cual como son

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento, #Abrira la carpeta de entrenamiento 
    target_size=(altura, longitud), #Redefine a valores predefinidos
    batch_size=batch_size, # Les asigna el bath size
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory( #Se hace lo anterior pero con la carpeta de Validacion
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

#----------------------Creacion de la Red Neuronal Convolucional------------------------------------------------

cnn = Sequential() # Se le dice que la red sera secuencial(varias capas apiladas entre ellas)
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))#Capa de la Primera Convolucion
cnn.add(MaxPooling2D(pool_size=tamano_pool)) #Capa de MaxPooling y tamano del filtro

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))#Segunda capa convolusional 
cnn.add(MaxPooling2D(pool_size=tamano_pool))#Segundo Maxpooling

cnn.add(Flatten()) #Se aplana la imagen (1 dimension con toda la informacion de la red neuronal)
cnn.add(Dense(256, activation='relu')) #Anade capa de 256 neuronas
cnn.add(Dropout(0.5)) #Se apaga el 50% de las neuronas en cada paso para evitar sobreentrenamiento y asi aprender caminos alternos
cnn.add(Dense(clases, activation='softmax'))#Segunda capa densa con 3 neuronas y la activacion de softmax para saber la probabilidad de la imagen respecto a las clases

#Parametros para optimizar el algoritmo
cnn.compile(loss='categorical_crossentropy',#Funcion de Perdida
            optimizer=optimizers.Adam(lr=lr),# Se optimiza con el gradiente Descendiente 
            metrics=['accuracy'])#Porcentaje de que tan bien esta aprendiendo la red

#-------------------------------------------------------------------------------------------------------------

#------------------------Entrenamiento del algoritmo----------------------------------------------------------
cnn.fit_generator(
    entrenamiento_generador, #imagenes de entrenamiento preprocesadas
    steps_per_epoch=pasos, #1000 pasos por epoca
    epochs=epocas, #20 epcoas
    validation_data=validacion_generador, 
    validation_steps=validation_steps) #200 pasos de validacion

target_dir = './modelo/'  
if not os.path.exists(target_dir):
  os.mkdir(target_dir) #Se genera la carpeta del modelo
cnn.save('./modelo/modelo.h5') #Se guarda el modelo en la carpeta de modelo
cnn.save_weights('./modelo/pesos.h5') #Se guardan los pesos en la carpeta modelo