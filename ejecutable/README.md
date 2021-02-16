CODIGO 
- Como primer paso  importamos las librerias
```sh
import cv2# libreria  open cv
import numpy as np#libreria numpy
```
-  Encendemos la camara 
```sh
cap = cv2.VideoCapture(0)#activamos camara
```
Espacio de color HSV
El espacio de color HSV (matiz, saturación, valor) es un modelo para representar el espacio de color similar al modelo de color RGB. Dado que el canal de matiz (H) modela el tipo de color, es muy útil en tareas de procesamiento de imágenes que necesitan segmentar objetos en función de su color. La variación de la saturación (S) puede entenderse como qué tan fuerte es el color que vemos; pasa de no saturada a representar sombras de gris y completamente saturada (sin componente blanco). El canal de valor (V) describe el brillo o la intensidad del color.
La mejor forma de encontrar un color que necesites es a través de esta imagen, donde se muestra la variación de colores, con H en el eje “x”, y S en el eje “y”, y V = 255.

[![N|Solid](https://github.com/KARENalejand/imagenes/blob/main/colores.png)](https://nodesource.com/products/nsolid)

Teniendo esto encuenta pasamos al tercer paso
- se estable una varible,con una matriz que asignara el color a buscar  
```sh
redBajo1 = np.array([80, 100, 20], np.uint8)
redAlto1 = np.array([90, 255, 255], np.uint8)
redBajo2=np.array([100, 100, 20], np.uint8)
redAlto2=np.array([120, 255, 255], np.uint8)
                  #(MATIZ,SATURACION,BRILLO
```
- Trasnformamos a hvs  y encontramos los rangos del color a detectar,hacemos la mask para poder mostrar el color que 
queremos detectar.
```sh
while True:
  ret,frame = cap.read()
  if ret==True:
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#tranformar a hsv
    maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)#se encuetran los rangos de la varible
    maskRed2 = cv2.inRange(frameHSV, redBajo2, redAlto2)## hacemos la mask y filtramos en la original
    maskRed = cv2.add(maskRed1, maskRed2)##se convierte en una sola mascara para detectar el color
    maskRedvis = cv2.bitwise_and(frame, frame, mask= maskRed)#va a mostrar el color real que queremos
```
-Como ultimo paso, visualizamos el video streming y con “s” se detiene la trasmisión
```sh
 cv2.imshow('frame', frame)
    cv2.imshow('maskRed', maskRed)
    cv2.imshow('maskRedvis', maskRedvis)
    if cv2.waitKey(1) & 0xFF == ord('s'):
      break
cap.release()
cv2.destroyAllWindows()
```
-Como podemos ver el programa detecta correctamente el color azul


[![N|Solid](https://github.com/KARENalejand/imagenes/blob/main/a1.jpeg)](https://nodesource.com/products/nsolid)

[![N|Solid](https://github.com/KARENalejand/imagenes/blob/main/a2.jpeg)](https://nodesource.com/products/nsolid)

Ahora para poder convertir este programa en un ejecutable  necesitamos instalar pyinstaller
```sh
pip install pyinstaller
```
#Importante
-el siguiente  codigo se debe ejecutar en simbolo del sistema  tanto para  windows como para linux   
```sh
pyinstaller --windowed --onefile COLORESpy
```
 -  Aqui crea un archivo ejecutable  con extención .exe en caso de windows y extención .spec en caso de linux, los cuales son encontrados en la carpeta  dist.
 
 - Estos archivos pueden ser difundido a diversas personas que no tengan instalado python  y pueden ejecutar el programa con solo dar dos clic en el ejecutable creado.
 
 Cabe mencionr que la extencion sera asignada desde el   sistema operativo que es creado. 
 
[![N|Solid](https://github.com/KARENalejand/imagenes/blob/main/ex.jpeg)](https://nodesource.com/products/nsolid)



