# CREAR EJECUTABLE 
- En este apartado se creará  un ejecutable   para  esto utilizaremos pyqt5 y auto-py-to-exe
# REQUISITOS
-Instalar
- pip install PyQt5
- pip install pyqt5-tools
- pip install auto-py-to-exe
# CREAR INTERFAZ GRAFICA 
-Escribimos  en el cmd, "designer" para abrir la siguiente ventana
 
 [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/designer.jpeg)]

- Elegimos "Main Window

[![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/designer2.jpeg)]


- En esta ventana colocaremos  lo necesario para crear una interfaz funcional ya sea, botones labels,  text edit etc.
- En este caso nuestra interfaz la crearemos de la siguiente manera:
[![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/interfaz.jpeg)]
- Teniendo ya nuestra interfaz  la guardaremos, este creara un archivo .ui 
- Teniendo nuestra interfaz guardada pasamos al codigo para hacer funcional  nuestra  interzas.
# CONVERTIR ARCHIVO .ui EN ARCHIVO .py
- Para esto escribe en el cmd, el siguiente código:
- pyuic5 nombre del arcvivo.ui -o nombre del archivo.py
- Teniendo  la conversió pasaremos al codigo
# CÓDIGO
 ```sh
 # IMPORTAMOS LIBRERIAS 
$ #!/usr/bin/env python3 # NO BORRAR ESTA PARTE HACE QUE FUNCIONE EL EJECUTABLE 
$ #-*- coding: utf8 -*-
$ import sys
$ import numpy.random.common
$ import numpy.random.bounded_integers
$ import numpy.random.entropy

$ from PyQt5 import  QtWidgets
$ print('Ejecutando aplicación')

$ #qtCreatorFile = "prueba1.ui" #Importamos el archivo .py de la interfaz
$ #Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
$ from prueba1 import Ui_MainWindow
$ global i # creamos una  variable global para utilizar en nuestro QComboBox
$ i = 1
$ class MyApp(QtWidgets.QMainWindow, Ui_MainWindow): # importamos clase
   $ def __init__(self):
     $   global i
      $  QtWidgets.QMainWindow.__init__(self)
      $  Ui_MainWindow.__init__(self)
      $  self.setupUi(self)
       $ self.encender.clicked.connect(self.detect) # funcion  para utilizar boton
       ##### definimos captura 
       $ def captura(self): 
       $  global i
       
       $ i = str(self.camara.currentText())# lo utilizamos para el QComboBox
       $ print(i)
       $ i = int(i)
       $ import cv2
       $ camera = cv2.VideoCapture(i) # tomará el valor de 1  ya sea uno o cero  esto servira para conectar una camara externa
       $ while(True):
       $     _, frame = camera.read()
                
       $     cv2.imshow("Camera", frame) # vizualizamos

       $     key = cv2.waitKey(1)
       $     if key & 0xFF == ord("q"): #se cierra co q
       $         break
       $     camera.release()
       $     cv2.destroyAllWindows()
   $  # definimos detect que servira para que el boton encender  
   $  # todo lo que este aqui dentro se realizara por el boton encender
   $ def detect(self):
    $    global i
       
    $    i = str(self.camara.currentText())
    $    print(i)
    $    i = int(i)
        ###################################importamos las librerias##############
     $   import numpy as np
     $   import pandas as pd
     $   import seaborn as sns
     $   from pylab import rcParams
     $   import matplotlib.pyplot as plt
     $   from matplotlib import rc
     $  from pandas.plotting import register_matplotlib_converters
     $   from sklearn.model_selection import train_test_split
     $   import urllib
     $   import os
     $   import csv
     $   import cv2
     $   import time
     $   from PIL import Image
     $   from keras_retinanet import models
     $   from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
     $   from keras_retinanet.utils.visualization import draw_box, draw_caption
     $   from keras_retinanet.utils.colors import label_color
     $   #################################### Cargamos el modelo##################################################
     $   ####################################debe estar en la carpeta /snapshots/##################################
     $   from keras.models import load_model
     $   from keras_retinanet import models

      $  model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0])
     $   print(model_path)

     $   model = models.load_model(model_path, backbone_name='resnet50')
     $   model = models.convert_model(model)

     $   labels_to_names = pd.read_csv('classes.csv', header=None).T.loc[0].to_dict()
     $   #########################################Función que realizaz la predicción#############################

     $   import skimage.io as io
     $   def predict(image):
     $       image = preprocess_image(image.copy())
     $       image, scale = resize_image(image)

     $       boxes, scores, labels = model.predict_on_batch(
     $        np.expand_dims(image, axis=0)
     $        )

     $       boxes /= scale

     $       return boxes, scores, labels

    $    #################################################Captura de Video#########################################
    $    camera = cv2.VideoCapture(i)# toma el valor de i que le demos desde QComboBox #############################
    $    camera_height = 500 #tamaño

     $   while(True):

     $       _, frame = camera.read()
            

    $        frame = cv2.flip(frame, 1)

    $        boxes, scores, labels = predict(frame)

   $         draw = frame.copy()


   $         for box, score, label in zip(boxes[0], scores[0], labels[0]):
   $             if score > 0.8: # detecta si es compatible un 0.8%
   $                 print(box) # imprime la caja
   $                 b = box.astype(int)
   $                 color = label_color(label)
   $                 draw_box(draw, b, color=color)
   $                 caption = "{} {:.3f}".format(labels_to_names[label], score)
   $                 draw_caption(draw, b, caption)
   $                 if label==0: # si la  etiqueta es igual a la clase 0 imprime  lo siguiente:
   $                     self.resultado.setText('se ha detectado : rasberry pi4')
   $                     si la  etiqueta es igual a la clase 1 imprime  lo siguiente:
   $                 if label==1:
   $                     self.resultado.setText('se ha detectado :protoboar')
                    

    $        # show the frame
    $        cv2.imshow("Test out", draw)# visualizamos  el video
            
    $        key = cv2.waitKey(1)
            
    $        # se cierra el video con q
    $        if key & 0xFF == ord("q"):
    $            break

    $    camera.release() # se apaga la camara
     $   cv2.destroyAllWindows()
###############esta parte  hace que se vizualice la interfaz######################################
$ if __name__ == '__main__':
$   app = QtWidgets.QApplication(sys.argv)
$    window = MyApp()
$    window.show()
$    sys.exit(app.exec_())
```
# CREAR ARCHIVO .EXE CON  auto-py-to-exe
- Escribimos en el cmd "auto-py-to-exe" este abrira la siguiente ventana:
[![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/PT-TO-EXE.jpeg)]
- Cargamos  nuestro código principal donde se hace la deteción de objetos
 [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/CARGAR.jpeg)]
 
 - Selecionamos  onefile y Window Based (hide the console), para crear un archivo .exe sin consola 
  [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/SIN%20CONSOLA.jpeg)] 

- Cargamos el icono que deseamos que tenga nuestra interfaz
- El archivo debe ser .ico  para convertir cualquier imagen en .ico  se utilizo la siguiente pagina: https://imagen.online-convert.com/es/convertir-a-ico

   [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/ICONO.jpeg)]  
   
- Cargamos los elementos necesarios  en el siguiente apartado:
- Primero cargaremos los folders:
- snapshots
- images
- keras-retinanet
 [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/FOLDER.jpeg)] 
 
 
   
        

       
       
       
       
       
       
       
       
       



