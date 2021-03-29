#!/usr/bin/env python3
#-*- coding: utf8 -*-
import sys
import numpy.random.common
import numpy.random.bounded_integers
import numpy.random.entropy

from PyQt5 import  QtWidgets
print('Ejecutando aplicación')

#qtCreatorFile = "prueba1.ui" # Nombre del archivo aquí.
#Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
from prueba1 import Ui_MainWindow
global i
i = 1

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        global i
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.encender.clicked.connect(self.detect)
        
        
    def captura(self):
        global i
       
        i = str(self.camara.currentText())
        print(i)
        i = int(i)
        import cv2
        camera = cv2.VideoCapture(i)
        while(True):
            _, frame = camera.read()
                
            cv2.imshow("Camera", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                break
            camera.release()
            cv2.destroyAllWindows()
    def detect(self):
        global i
       
        i = str(self.camara.currentText())
        print(i)
        i = int(i)
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from pylab import rcParams
        import matplotlib.pyplot as plt
        from matplotlib import rc
        from pandas.plotting import register_matplotlib_converters
        from sklearn.model_selection import train_test_split
        import urllib
        import os
        import csv
        import cv2
        import time
        from PIL import Image
        from keras_retinanet import models
        from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
        from keras_retinanet.utils.visualization import draw_box, draw_caption
        from keras_retinanet.utils.colors import label_color
        # Cargamos el modelo
        # debe estar en la carpeta /snapshots/
        from keras.models import load_model
        from keras_retinanet import models

        model_path = os.path.join('snapshots', sorted(os.listdir('snapshots'), reverse=True)[0])
        print(model_path)

        model = models.load_model(model_path, backbone_name='resnet50')
        model = models.convert_model(model)

        labels_to_names = pd.read_csv('classes.csv', header=None).T.loc[0].to_dict()
        # Función que realizaz la predicción

        import skimage.io as io
        def predict(image):
            image = preprocess_image(image.copy())
            image, scale = resize_image(image)

            boxes, scores, labels = model.predict_on_batch(
             np.expand_dims(image, axis=0)
             )

            boxes /= scale

            return boxes, scores, labels

        # Captura de Video
        camera = cv2.VideoCapture(i)
        camera_height = 500

        while(True):

            _, frame = camera.read()
            

            frame = cv2.flip(frame, 1)

            boxes, scores, labels = predict(frame)

            draw = frame.copy()


            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                if score > 0.8:
                    print(box)
                    b = box.astype(int)
                    color = label_color(label)
                    draw_box(draw, b, color=color)
                    caption = "{} {:.3f}".format(labels_to_names[label], score)
                    draw_caption(draw, b, caption)
                    if label==0:
                        self.resultado.setText('se ha detectado : rasberry pi4')
                    if label==1:
                        self.resultado.setText('se ha detectado :protoboar')
                    

            # show the frame
            cv2.imshow("Test out", draw)
            detec=label
            key = cv2.waitKey(1)
            

           
                

            # quit camera if 'q' key is pressed
            if key & 0xFF == ord("q"):
                break

        camera.release()
        cv2.destroyAllWindows()







   


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
        
