# CLASIFICADOR-DE-IMAGENES

###PRIMER PASO

- Creamos una carpeta con el nombre lcd y otra con el nombre perro.
[![N|Solid](https://github.com/KARENalejand/imagenes/blob/main/f.png)](https://nodesource.com/products/nsolid)

- Dentro de dichas carpetas estarán las imágenes que se utilizarán para el
entrenamiento.
[![N|Solid](https://github.com/KARENalejand/imagenes/blob/main/CARPETAS.png)](https://nodesource.com/products/nsolid)

### Código 
- Teniendo  completo el primer paso pasamos a  realizamos el siguiente codigo

- primero importamos las  librerias 

```sh
$ import cv2
$ import numpy as np
$ import os
$ import matplotlib.pyplot as plt
```

- Luego cargamos las fotos  de las carpetas especificas
```sh
$ lcd_folder_path="C:/Users/kv460/Documents/lcd/lcd"
$ lcd=[]
$ img_size=150
$ for img in os.listdir(lcd_folder_path):
    $ img = cv2.imread(os.path.join(lcd_folder_path,img))
    $ #img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    $img_resize= cv2.resize(img,(img_size,img_size))
    $lcd.append(img_resize)
```
En esta imagen se explica cada parte de  ese texto
[![N|Solid](https://github.com/KARENalejand/imagenes/blob/main/parte2.png)](https://nodesource.com/products/nsolid)

- vizualizamos tamaño de la imagen

```sh
$ lcd = np.array(lcd)
$ print(lcd.shape)
```
- Se realiza exactamente lo mismo, pero ahora para la carpeta de perro

```sh
$ ####cargamos las fotos  de las carpetas especificas### 
$ perro_folder_path="C:/Users/kv460/Documents/lcd/perro"
$ perro=[]
$ img_size=150
$ for img in os.listdir(perro_folder_path):
    $ img = cv2.imread(os.path.join(perro_folder_path,img))
    $ #img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    $ img_resize= cv2.resize(img,(img_size,img_size))
    $ perro.append(img_resize)
    
$ perro = np.array(perro)
$ print(perro.shape)
```
- Graficamos las imagenes
```sh
$ print(lcd[4].shape)
$ plt.figure()
$ plt.imshow(np.squeeze(lcd[2]))
$ plt.colorbar()
$ plt.grid(False)
$ plt.show()
######
$ print(perro[2].shape)
$ plt.figure()
$ plt.imshow(np.squeeze(perro[2]))
$ plt.colorbar()
$ plt.grid(False)
$ plt.show()
```

[![N|Solid](https://github.com/KARENalejand/imagenes/blob/main/parte%203.png)](https://nodesource.com/products/nsolid)

- Se juntarán las imágenes con una concatenación
```sh
$ images=np.concatenate([lcd,perro])
$ #convertir a un arreglo 
$ Images=np.array(images)
$ print(len(images)) #imprimecuantas imagenes hay en total
```
- En este momento etiquetas todas nuestras imágenes teniendo en cuenta, cuantas imágenes hay en cada carpeta en este caso en lcd hay 41 y en la carpeta perro hay81

```sh
$ etiquetas_lcd=np.repeat(0,41)#lcd se reconoce con elnumero 0
$ print(len(etiquetas_lcd))
$ print(etiquetas_lcd)
#################etiquetamos##############
$ etiquetas_perro=np.repeat(1,81)#perro se reconoce con elnumero 1
$ print(len(etiquetas_perro))
$ print(etiquetas_perro)
```
- Ahora damos el nombre a nuestras clases teniendo en cuenta que lcd fue nuestra primera etiqueta y perro la segunda, por lo cual deben ser colocadas en ese orden.
Posterior a esto concatenamos las etiquetas
```sh
$ class_names=['lcd','perro']
$ #concatenamos clases
$ labels=np.concatenate([etiquetas_lcd,etiquetas_perro])
$ print(len(labels))
$ print(labels)
```
- Graficamos las primeras 100 imágenes

```sh
$ #arreglo
$ Labels=np.array(labels)
$ print(Labels.shape)
$ plt.figure(figsize=(10,10))
$ for i in range(100):#GRAFICA 100 FIGURAS
    $ plt.subplot(10,10,i+1)
    $ plt.xticks([])
    $ plt.yticks([])
    $ plt.grid(False)
    $ plt.imshow(Images[i])
    $ #, cmap=plt.cm.binary
   $ plt.xlabel(class_names[Labels[i]])
$ plt.show()
```
[![N|Solid](https://github.com/KARENalejand/imagenes/blob/main/etiquetas.png)](https://nodesource.com/products/nsolid)

- importamos las librerias de Tensorflow
```sh
$ import tensorflow as tf
$ from tensorflow import keras
$ from tensorflow.keras import datasets, layers, models
$ import tensorflow.keras.optimizers as Optimizer

$ # Helper libraries
$ import numpy as np
$ import matplotlib.pyplot as plt
$print(tf.__version__)
```
- Se realiza el entrenamiento con 30 iteraciones
```sh
$ model = keras.Sequential([
   $ keras.layers.Flatten(input_shape=(150, 150,3)),
    $ keras.layers.Dense(128, activation='relu'),
    
    $ keras.layers.Dense(2, activation='softmax'),
    
])
$model.compile(optimizer='adam',
              $loss='sparse_categorical_crossentropy',
              $ metrics=['accuracy'])
 $model.fit(Images, Labels, epochs=3)
$ trained=model.fit(Images, Labels, epochs=30)
```

[![N|Solid](https://github.com/KARENalejand/imagenes/blob/main/entrenamiento.png)](https://nodesource.com/products/nsolid)

- Para comprobar que quedo adecuadamente cargamos una imagen y la visualizamos
```sh
 $#imagen de interner se debe cargar 
$ img=cv2.imread('p.jpeg')
$ img_cvt=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
$ cv2.imshow('imagen',img)
```


- Se hace un cambio de tamaño en la imagen para que quede de 150*150 con 3 canales
```sh
$ img2=img_cvt
$ img2=cv2.resize(img2,(img_size,img_size))
$ print(img2.shape)
$ img2=(np.expand_dims(img2,0))
$ print(img2.shape)
```
- Se realiza la predicción para saber cómo la clasifica si como una lcd o un perro
```sh
predictions_single=model.predict(img2)
print(predictions_single)
print(np.sum(predictions_single))
print(np.argmax(predictions_single))
print(class_names[np.argmax(predictions_single)])
```
[![N|Solid](https://github.com/KARENalejand/imagenes/blob/main/predicion.png)](https://nodesource.com/products/nsolid)


###Resultados 
- Podemos notar que pudo clasificar la imagen como un perro.
[![N|Solid](https://github.com/KARENalejand/imagenes/blob/main/prueba.png)](https://nodesource.com/products/nsolid)




