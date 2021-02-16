import cv2# libreria  open cv
import numpy as np#libreria numpy
cap = cv2.VideoCapture(0)#activamos camara
redBajo1 = np.array([80, 100, 20], np.uint8)#se estable una varible,con una matriz
# que asignara elclora buscar
redAlto1 = np.array([90, 255, 255], np.uint8)
redBajo2=np.array([100, 100, 20], np.uint8)
redAlto2=np.array([120, 255, 255], np.uint8)
                  #(MATIZ,SATURACION,BRILLO)
while True:
  ret,frame = cap.read()
  if ret==True:
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#tranformar a hsv
    maskRed1 = cv2.inRange(frameHSV, redBajo1, redAlto1)#se encuetran los rangos de la varible
    maskRed2 = cv2.inRange(frameHSV, redBajo2, redAlto2)## hacemos la mask y filtramos en la original
    maskRed = cv2.add(maskRed1, maskRed2)##se convierte en una sola mascara para detectar el color
    maskRedvis = cv2.bitwise_and(frame, frame, mask= maskRed)#va a mostrar el color real que queremos
    
    cv2.imshow('frame', frame)
    cv2.imshow('maskRed', maskRed)
    cv2.imshow('maskRedvis', maskRedvis)
    if cv2.waitKey(1) & 0xFF == ord('s'):
      break
cap.release()
cv2.destroyAllWindows()
