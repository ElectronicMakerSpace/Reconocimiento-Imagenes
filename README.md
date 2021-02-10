# Reconocimiento optico
[![N|Solid](https://github.com/KARENalejand/Reconocimiento-ptico-/blob/main/descarga.jpeg)]

###  INSTALACIÓN DE Tesseract – OCR

>Como primer paso vamos a instalar Tesseract – OCR del siguiente link
>https://github.com/UB-Mannheim/tesseract/wiki,
>una vez descargado, ejecutamos el instalador.
>Aceptamos los términos y en la sección de Selección de componentes
>vamos a ver que por defecto se instalan los datos del idioma inglés.
>Si deseamos otros idiomas vamos a expandir Adicional language data, 
>en donde podremos elegir aquellos que deseemos instalar.
>Una vez finalizada la instalación vamos a dar clic en Terminar.

### Codigo
[![N|Solid](https://github.com/KARENalejand/Reconocimiento-ptico-/blob/main/image3.jpeg)]


Teniendo instalado Tesseract – OCR  procedemos a desarrollar el siguiente codigo:

- primero importamos las librerias correspondientes
```sh
$ import cv2
$import pytesseract
```

- Para evita problemas donde el codigo no encuentra el programa instalado Vamos a copiar y pegar en el script de nuestro programa
```sh
$ pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'

```
- Leemos la imagen de la cual vamos a reconocer el texto.
```sh
$ prueba1 = cv2.imread('tex.jpg')
```
- Para poder emplear el reconocimiento óptico de caracteres usamos pytesseract.image_to_string y entre paréntesis la variable en donde está asignada la imagen.
```sh
$ text = pytesseract.image_to_string(prueba1,lang='spa')
```

- Se va a imprimir el texto extraído de la imagen.
```sh
$ print('Texto: ',text)
``` 
- Vamos a visualizar la imagen, esperar a que una tecla sea presionada para que acabe el proceso y se cierren las ventanas.
```sh
$ #vizualizamos la imagen     
$cv2.imshow('Image',prueba1)
$#cerramos y borramos todo
$cv2.waitKey(0)
$cv2.destroyAllWindows()
``` 
### Pruebas de funcionamiento

[![N|Solid](https://github.com/KARENalejand/Reconocimiento-ptico-/blob/main/funcionamiento.jpeg)]



























