import cv2
import pytesseract
#donde esta el programa instalado
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'
#leer imagen
prueba1 = cv2.imread('tex.jpg')
#reconocimiento optico de caracteres esto  se almacena
#en la variable text. y aqui imprimimos lo que se encuentre 
text = pytesseract.image_to_string(prueba1,lang='spa')
print('Texto: ',text)
 #vizualizamos la imagen     
cv2.imshow('Image',prueba1)
#cerramos y borramos todo
cv2.waitKey(0)
cv2.destroyAllWindows()
