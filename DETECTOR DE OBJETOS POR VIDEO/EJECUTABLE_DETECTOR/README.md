# CREAR EJECUTABLE 
- En este apartado se creará  un ejecutable, para esto utilizaremos pyqt5 y auto-py-to-exe
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
- Teniendo nuestra interfaz guardada pasamos al código para hacer funcional  nuestra  interzas.
# CONVERTIR ARCHIVO .ui EN ARCHIVO .py
- Para esto escribiremos en el cmd, el siguiente código:
- pyuic5 nombre del arcvivo.ui -o nombre del archivo.py
- Teniendo  la conversió pasaremos al código
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
       
###############esta parte  hace que se vizualice la interfaz######################################
$ if __name__ == '__main__':
$   app = QtWidgets.QApplication(sys.argv)
$    window = MyApp()
$    window.show()
$    sys.exit(app.exec_())
```
- El código completo lo encuentras en la parte superior con el nombre de "prueba2"
-
# CREAR ARCHIVO .EXE CON  auto-py-to-exe
- Escribimos en el cmd "auto-py-to-exe", este abrirá la siguiente ventana:
[![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/PT-TO-EXE.jpeg)]
- Cargamos  nuestro código principal, donde se realiza la deteción de objetos
 [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/CARGAR.jpeg)]
 
 - Selecionamos  onefile y Window Based (hide the console), para crear un archivo .exe sin consola 
  [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/SIN%20CONSOLA.jpeg)] 

- Cargamos el icono que deseamos que tenga nuestra interfaz
- El archivo debe ser .ico 
- Para convertir cualquier imagen en .ico , se utilizo la siguiente pagina: https://imagen.online-convert.com/es/convertir-a-ico

   [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/ICONO.jpeg)]  
   
- Cargamos los elementos necesarios  en el siguiente apartado:
- Primero cargaremos los folders:
- snapshots
- images
- keras-retinanet


 [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/FOLDER.jpeg)] 
 
 - También cargaremos los archivos:
 - annotations 
 - annotations_test 
 - classes
 - prueba1.py( este archivo es  el archivo .ui convertido a .py)
 - Esto lo logramos en el siguiente apartardo
 
 [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/ARCHIVOS.jpeg)] 
 
 - Es necesario tener todos estos archivos de lo contrario  el ejecutable no funcionara
 
 - Finalmente compilamos 
 - presionanos el siguiente boton:
 
 [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/EJECUTAR.jpeg)] 
   
   - Al termina de compila se creara una carpeta donde  se encontrara el archivo .exe
    [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/CARPETA_EXE.jpeg)] 
   
   - El archivo se vera de la siguiente manera:
    
    [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/EXE_APP.jpeg)] 
   
    
   

 
 
 
 
 
 
 
 
 
 
 - Tambien se recomienda uqe primero se hagan pruebas con un archivo con consola para ver si sale algunos   errores 
 -  El error  que te puedes encontra seria el siguiente 
 
 
 
 
 
 
   
        

       
       
       
       
       
       
       
       
       



