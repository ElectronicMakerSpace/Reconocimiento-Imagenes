# Detección de objetos en video usando el modelo Retinanet

- COMO PRIMER PASO
# DESCARGAMOS  labelImg
- PARA ESTO CLONAMOS  EL SIGUIENTE REPOSITORIO
git clone https://github.com/tzutalin/labelImg.git
- cd labelImg
# DESCARGAR IMAGENES 
- COMO SEGUNDO JUNTAMOS IMAGENES EN FORMATO PNG DE LOS OBJETOS ESPECIFICOS QUE QUEREMOS DETECTAR EN ESTE CASO SE  DESCARGARON 700 FOTOS DE RASBPERRY PI 4 Y 500 IMAGENES DE PROTOBOART.
- POSTADA: Para realizar un mejor entrenamiento se recomienda  tener 1500 imagenes como minimo  de cada clase a detectar.
- Despues de tener nuestras imagenes procedemos a etiquetarlas con labelImg
 
 [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/labels.jpeg)]
 
 Este programa creara un archivo XML donde se encuentran las coordenadas del objeto a detectar y el nombre de la clase
  [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/label2.jpeg)]
 - Terminando de  etiquetar todas las imagenes  convertiremos los archivos  XML a  archivos CSV  apoyandonos de jupyter notebook.
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
 Con este programa se ejecutaran 3 archivos en formato CSV clases contendra los nombre de los objetos a detectar, annotations.csv contendra los archicvos que se utilizaran para  entremaniento y annotations_test.csv contendra los archivos que se utilizaran para pruebas.
 
 [![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/cvs.jpeg)]

# ENTRENAMIENTO
- Para este proceso usaremos google colab  para que sea mas rapido el proceso
- Abrimos un nuevo archivo y  configuramos un  entorno de ejecucion GPU.

[![N|Solid](https://github.com/ElectronicMakerSpace/Reconocimiento-Imagenes/blob/main/DETECTOR%20DE%20OBJETOS%20POR%20VIDEO/im%C3%A1genes%20para%20%20readme/gpu.jpeg)]

 
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
 








