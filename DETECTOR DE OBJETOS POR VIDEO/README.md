# DETECCIÓN DE OBJETOS EN VIDEO, USANDO EL MODELO Retinanet


# DESCARGAMOS  labelImg
- PARA ESTO CLONAMOS  EL SIGUIENTE REPOSITORIO
git clone https://github.com/tzutalin/labelImg.git
- cd labelImg
# DESCARGAR IMAGENES 
- COMO SEGUNDO PASO JUNTAMOS IMAGENES EN FORMATO.PNG DE LOS OBJETOS ESPECIFICOS QUE QUEREMOS DETECTAR EN ESTE CASO SE  DESCARGARON 700 FOTOS DE RASBPERRY PI 4 Y 500 IMAGENES DE PROTOBOARD.
- POSDATA: Para realizar un mejor entrenamiento se recomienda  tener 1500 imagenes como minimo  de cada clase a detectar.
- Después de tener nuestras imagenes, procedemos a etiquetarlas con labelImg
 
 [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/labels.jpeg)]
 
 Este programa creará un archivo .XML donde se encuentran las coordenadas del objeto a detectar y el nombre de la clase
  [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/label2.jpeg)]
  # Archivos XML a archivos CSV
 - Terminando de  etiquetar todas las imagenes, convertiremos los archivos  .XML a  archivos .CSV  apoyandonos de jupyter notebook.
 # CÓDIGO
 ```sh
 # importamoslibrerias
$ import os
$ import glob
$ import pandas as pd
$ import xml.etree.ElementTree as ET
```
 ```sh
 $ def xml_to_csv(path):
    $ xml_list = []
    $ for xml_file in glob.glob(path + '/*.xml'):
        $ tree = ET.parse(xml_file)
        $ root = tree.getroot()

        $ for member in root.findall('object'):
            $ value = ('images/'+root.find('filename').text,
                     $ int(member[4][0].text),
                     $ int(member[4][1].text),
                     $ int(member[4][2].text),
                     $ int(member[4][3].text),
                     $member[0].text,
                     )
            $ xml_list.append(value)
    $ column_name = ['filename', 'xmin','ymin','xmax', 'ymax', 'class']
    $ xml_df = pd.DataFrame(xml_list, columns=column_name)
    $ return xml_df


$ image_path = os.path.join(os.getcwd(), 'images/')
$ dataset_df = xml_to_csv(image_path)

$print('Completed')
```
 ```sh
 # MOSTRAR DATASET 
$ import skimage.io as io
$ import matplotlib.pyplot as plt
$ from sklearn.model_selection import train_test_split
$ import matplotlib.patches as patches

$ %matplotlib inline

$ def showObjects(image_df):

    $ img_path = image_df.filename

    $ image = io.imread(img_path)
    $ draw = image.copy()
    
    $ # Create figure and axes
    $ fig,ax = plt.subplots(1)
    $ ax.imshow(draw)
    $ rect = patches.Rectangle((image_df.xmin,image_df.ymin),image_df.xmax-image_df.xmin,image_df.ymax-image_df.ymin,linewidth=1,edgecolor='r',facecolor='none')

    $ plt.axis('off')
    $ ax.add_patch(rect)
    $ plt.show()

   $showObjects(dataset_df.iloc[2])#muestra las imagenes con el objeto detectado para saber que realmente estamos trabajando en el directoro deseado  
 ```
 ```sh 
 # DIVIDIR DATASET
train_df, test_df = train_test_split(
  dataset_df, #dataset completo
  test_size=0.2,# se utiliza el 20% para  pruebas y el 80% para entrenamiento 
  random_state=2
)

# CREAR ARCHIVOS CSV
$ train_df.to_csv('annotations.csv', index=False, header=None)# entrenamiento
$ test_df.to_csv('annotations_test.csv', index=False, header=None)# pruebas

$ "classes = set(['protoboar','Raspberry pi4'])\n",

$ with open('classes.csv', 'w') as f:
    $ for i, line in enumerate(sorted(classes)):
        $ f.write('{},{}\n'.format(line,i))
 ```
 Con este programa se ejecutaran 3 archivos en formato .CSV , el archivo clases contendra los nombre de los objetos a detectar, annotations.csv contendra los archicvos que se utilizaran para  entremaniento y annotations_test.csv contendra los archivos que se utilizaran para pruebas.
 
 [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/cvs.jpeg)]

# ENTRENAMIENTO
- Para este proceso usaremos google colab  para que sea más rápido el proceso
- Abrimos un nuevo archivo y  configuramos un  entorno de ejecución GPU.

[![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/gpu.jpeg)]

# CÓDIGO DE ENTRENAMIENTO
 
 ```sh 
#Como primer paso clonamos el repositorio de keras-retinanet
$!git clone https://github.com/DavidReveloLuna/keras-retinanet.git
 $ # instalamos los siguientes paquetes
$!pip install keras==2.3.1
$ !pip install tensorflow==2.1
# Nos dirigimos a la siguiente carpeta
cd keras-retinanet/
# Instalación y configuración de keras-retinet
!pip install .
!python setup.py build_ext --inplace

# Montar tu drive
from google.colab import drive
drive.mount('/content/drive')
```
- Al ejecutar "from google.colab import drive....." se   mostrará  el siguiente link:
- Dar clicK  en el  link para  ingresar  a google drive 
[![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/drive.jpeg)]

- Al ingresar a google drive, aparecerá el siguente código el cual deberas ingresar en el recuadro que apareció  en google colab.

[![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/codigo%20drive.jpeg)]

# IMPORTANTE
DEBES CARGAR LOS ARCHIVOS CSV Y LAS IMAGENES ETIQUETADAS EN GOOGLE DRIVE PARA PODER REALIZAR ESTE PROCESO
# CONTINUAMOS:
- Al montar nuestro drive escribimos las siguientes lineas de código para  copiar los archivos de entrenamiento que necesitamos

```sh 
$ # Copiamos los archivos que necesitamos para el entrenamiento
$ # Asegúrate de reemplazar la dirección con tu propio path (/content/drive/My Drive/Desarrollos/Hand Detection/) 
$ !cp -r "/content/drive/My Drive/prueba5/images" "/content/keras-retinanet"
$ !cp -r "/content/drive/My Drive/prueba5/annotations.csv" "/content/keras-retinanet"
$ !cp -r "/content/drive/My Drive/prueba5/annotations_test.csv" "/content/keras-retinanet"
$ !cp -r "/content/drive/My Drive/prueba5/classes.csv" "/content/keras-retinanet" 
```
#TENIENDO  LOS ARCHIVOS PROCEDEMOS CON EL CÓDIGO  
```sh 
# importamos librerias
$ import numpy as np
$ import pandas as pd
$ import seaborn as sns
$ from pylab import rcParams
$ import matplotlib.pyplot as plt
$ from matplotlib import rc
$ from pandas.plotting import register_matplotlib_converters
$ from sklearn.model_selection import train_test_split
$ import urllib
$ import os
$ import csv
$ import cv2
$ import time
$ from PIL import Image

$ from keras.models import load_model
$ from keras_retinanet import models
$ from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
$ from keras_retinanet.utils.visualization import draw_box, draw_caption
$ from keras_retinanet.utils.colors import label_color
```
# DESCARGAMOS  EL MODELO PREENTRENADO resnet50_coco
```sh 
URL_MODEL = 'https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5'
urllib.request.urlretrieve(URL_MODEL, './snapshots/model.h5')
# este modelo se grabara en la carpeta snapshots
```

# Es importante correr la siguiente linea de codigo, si no, no se llevara acabo  el entrenamiento.
```sh 
!chmod 777 keras_retinanet/bin/*
```
# SE REALIZA EL ENTRENAMIENTO 

```sh 
!keras_retinanet/bin/train.py --freeze-backbone --random-transform --weights snapshots/model.h5 --batch-size 8 --steps 200 --epochs 50 csv annotations.csv classes.csv
#Aqui se realizará el entrenamiento , aqui especificamos el numero de epocas , batch y eteps.
#Cabe mencionar que este puede tardar horas  o dias dependiendo   del nuemero de batch-size,steps,epochs y el internet ,También cabe mencionar que debes cuidar este  proceso para que no haya sobre entrenamiento.
```
- A si se verá el entrenamiento de la ultima epoca
[![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/E.jpeg)]
 
 - Todas estas epocas  se guardarán en la carpeta snapshots y se deberá de descargar el ultimo o el penultimo entrenamiento, ya que son los que mas han aprendido
 # REALIZAR  PRUEBA DE FUNCIONAMIENTO
 - Con el siguiente código realizamos una prueba, para ver que el entrenamiento funciona de manera adecuada.
 
```sh 
$ model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0])
$ print(model_path)
$ model = models.load_model(model_path, backbone_name='resnet50')

$ labels_to_names = pd.read_csv('classes.csv', header=None).T.loc[0].to_dict()
$ test_df = pd.read_csv("annotations_test.csv")
$ test_df.head()
$ # Obtenemos la predicción del modelo: boxes, scores, labels
$ import skimage.io as io

$ def predict(image):
  $ image = preprocess_image(image.copy())
  $ image, scale = resize_image(image)

  $ boxes, scores, labels = model.predict_on_batch(
    $ np.expand_dims(image, axis=0)
  $ )

  $ boxes /= scale

  $ return boxes, scores, labels
  
 $ # Mostramos los objetos encontrardos en la imagen
 $# Se toman encuenta sólo los objetos que tienen asociada una probabilidad mayor a umbralScore
$ umbralScore = 0.8

$ def draw_detections(image, boxes, scores, labels):
  $ for box, score, label in zip(boxes[0], scores[0], labels[0]):
    $ if score < umbralScore:
       $ break

    $ color = label_color(label)

    $ b = box.astype(int)
    $ draw_box(image, b, color=color)

   $ caption = "{} {:.3f}".format(labels_to_names[label], score)
   $  draw_caption(image, b, caption)
   
  $ # Recorremos todo el dataFramee de test para revisar las predicciones
$ for index, row in test_df.iterrows():
 $ print(row[0], index)
 $ image = io.imread(row[0])

$  boxes, scores, labels = predict(image)

 $ draw = image.copy()
 $ draw_detections(draw, boxes, scores, labels)

 $ plt.axis('off')
 $ plt.imshow(draw)
 $ plt.show()
 ```
 - SI TODO SALE BIEN, SE VERÁ DE LA SIGUIENTE MANERA:



 [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/prueba.jpeg)]
 
 
 # CREAR DETECTOR DE COMPONENTES ELECTRONICÓS
 - Creamos  una carpeta donde deben estar los siguientes archivos
 - annotations.csv
 - annotations_test.csv
 - clases.csv
 - Una carpeta llamada snapshots(contendrá el modelo entrenado que descargamos)
 - Carpeta images(contendrá  imagenes con sus debidas etiquetas)


 [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/carpeta.jpeg)]

# CREAR ENTORNO VIRTUAL EN ANACONDA 
```sh 
 $ conda create -n entorno anaconda python=3.7.7
 $ conda activate entorno
 $ conda install ipykernel
$ conda install tensorflow-gpu==2.1.0 cudatoolkit=10.1
$ pip install tensorflow==2.1.0
$ pip install jupyter
$ pip install keras==2.3.1
$ pip install numpy scipy Pillow cython matplotlib scikit-image opencv-python h5py imgaug IPython[all]
```
# ABRIMOS JUPYTER NOTEBOOK
- Aquí se llevará acabo la deteción de objetos en tiempo real atravez de video streaming
- Creamos  un documento nuevo y empezamos con el código.
```sh 
# Empezamos  clonando algunas librerias ya  que es nuevo el entorno 
$ !git clone https://github.com/DavidReveloLuna/keras-retinanet.git
$ cd keras-retinanet/
$ !pip install .
$ !python setup.py build_ext --inplace
#luego importamos las  LIBRERIAS
$ import numpy as np
$ import pandas as pd
$ import seaborn as sns
$  pylab import rcParams
$ import matplotlib.pyplot as plt
$ from matplotlib import rc
$ from pandas.plotting import register_matplotlib_converters
$ from sklearn.model_selection import train_test_split
$ import urllib
$ import os
$ import csv
$ import cv2
$ import time
$ from PIL import Image

$ from keras_retinanet import models
$ from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
$ from keras_retinanet.utils.visualization import draw_box, draw_caption
$ from keras_retinanet.utils.colors import label_color
$ from keras_retinanet import models
$ from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
$ from keras_retinanet.utils.visualization import draw_box, draw_caption
$ from keras_retinanet.utils.colors import label_color
$ #CARGAMOS EL MODELO Y LAS EIQUETAS QUE ESTAN EN LOS ARCHIVOS
$ from keras.models import load_model
$ from keras_retinanet import models

$ model_path = os.path.join('snapshots', sorted(os.listdir('snapshots/'), reverse=True)[0])
$ print(model_path)

$ model = models.load_model(model_path, backbone_name='resnet50')
$ model = models.convert_model(model)

$ labels_to_names = pd.read_csv('classes.csv', header=None).T.loc[0].to_dict()

# Función que realizaz la predicción
# DEVUELVE LOS OBJETOS DE INTERES

$ import skimage.io as io

$ def predict(image):
    $ image = preprocess_image(image.copy())# PROCESA LA IMAGEN
    $ image, scale = resize_image(image)#

    $ boxes, scores, labels = model.predict_on_batch( # DEVUELVE REGIONES DE INTERÉS, LOS PUNTAJE Y LAS ETIQUETAS
     $ np.expand_dims(image, axis=0)
     )

   $  boxes /= scale

    $ return boxes, scores, label
    
```
    
   # SE REALIZA LA CAPTURA DE VIDEO
 ```sh
    
$ camera = cv2.VideoCapture(0) # SE INICIA LA CAMARA
$ camera_height = 500 #tamaño

$ while(True):

 $   _, frame = camera.read()
    

  $  frame = cv2.flip(frame, 1)

 $   boxes, scores, labels = predict(frame) #

  $  draw = frame.copy()


 $  for box, score, label in zip(boxes[0], scores[0], labels[0]):
 $       if score > 0.8: #si es un 0.8% compatible se detecta el objeto
 $           print(box) # se dibuja la caja
 $           b = box.astype(int)
 $           color = label_color(label)
 $           draw_box(draw, b, color=color)
 $          caption = "{} {:.3f}".format(labels_to_names[label], score)
 $          draw_caption(draw, b, caption)
            

    # show the frame
  $  cv2.imshow("Test out", draw)# se visualiza el video


  $  key = cv2.waitKey(1)

  
   $ if key & 0xFF == ord("q"): #presiona q para detener el video
    $    break

camera.release()
cv2.destroyAllWindows()$ borra información de la camara

```
# RESULTADOS
 [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/RESULTADO%201.jpeg)]
 
[![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/RESULTADO2.jpeg)]

# DETECTAR IMAGENES
- Si se desea detectar imagenes,solo se omite la parte de captura de video y se  coloca el siguiente  código.
```sh
# IMAGENES
img=cv2.imread('809.png')#SE CARGA LA IMAGEN 
cv2.imshow('imagen',img) # SE VIZUALIZA
boxes,scores,labels=predict(img)
draw=img.copy()

for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score > 0.8:
            print(box)
            b = box.astype(int)
            color = label_color(label)
            draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
            cv2.imshow("imagen",draw)# SE DIBUJA EL RECTANGULO EN CASO  DE DETECTAR EL OBJETO DESEADO
           
            print('se ha detectado')
            detec=label
            if label==0:
                print('rasberry pi4')
            if label==1:
                print('protoboar')
                cv2.waitKey(0) # SIRVE PARA CERRAR lA IMAGEN
            break 
        else: # SI NO  DETECTATA NADA  EN LA IMAGEN IMPRIME " NO  SE DETECTO  INTENTE DE NUEVO"
            cv2.imshow('imagen',img)# VIZUALIZA LA IMAGEN
            print('no se detecto intente de nuevo')
            cv2.waitKey(0)# CIERRA LA IMAGEN
            break
   ```
   # RESULTADOS
    
[![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/RESULTADO3.jpeg)]
 




 




 








