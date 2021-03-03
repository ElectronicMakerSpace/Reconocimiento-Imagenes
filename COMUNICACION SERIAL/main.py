import sys #librerias
from PyQt5.QtWidgets import QMainWindow,QApplication
from GUI import *#importamos la gui
from customSerial import customSerial #comunicacion 

class MiApp(QMainWindow):#ventana principal
	def __init__(self):
		super().__init__()
		self.ui = Ui_MainWindow()#clase que almacena la informacion grafica
		self.ui.setupUi(self)#se configura en la interface

		#Serial
		self.serial = customSerial()# obtiene la logita del otro archivo
                #actualiza la lista de batudlist
		self.ui.BaudList.addItems(self.serial.baudratesDIC.keys())
		self.ui.BaudList.setCurrentText('9600')#velocidad por defecto
		self.update_ports()#se actualizan los puertos
		
		#se  definen los botones para cuando sean apretados cumpla una funcion
		self.ui.connectBtn.clicked.connect(self.connect_serial)
		self.ui.sendBtn.clicked.connect(self.send_data)
		self.ui.updateBtn.clicked.connect(self.update_ports)
		self.ui.clearBtn.clicked.connect(self.clear_terminal)
		self.serial.data_available.connect(self.update_terminal)

	def update_terminal(self,data):
		self.ui.Terminal.append(data)##AÑADE EL DATO
        #conectar
	def connect_serial(self):
		if(self.ui.connectBtn.isChecked()):# si presiona conecta
			port = self.ui.portList.currentText()#
			baud = self.ui.BaudList.currentText()#lee baudios
			self.serial.serialPort.port = port #accede al serial port
			self.serial.serialPort.baudrate = baud#accedea baud
			self.serial.connect_serial()#conecta al serial connect
			#Si se logra conectar
			if(self.serial.serialPort.is_open):#si se conecta
				self.ui.connectBtn.setText('DESCONECTAR')#el boton
				#conectar muestra el mensaje desconectar
				#print("Me conecté")

			#No se logró conectar	
			else:
				#print("No me conecte")
				self.ui.connectBtn.setChecked(False)#
			
		else:
			#print("Desconectarme")
			self.serial.disconnect_serial()#se desconecta
			self.ui.connectBtn.setText('CONECTAR')#se pone el mensaje conectar en el boton
        #enviar
	def send_data(self):
		data = self.ui.input.text()#lee datos del objeto input
		self.serial.send_data(data)#se envia al objeto serial
        #actualizar puertos
	def update_ports(self):
		self.serial.update_ports()
		self.ui.portList.clear()#
		self.ui.portList.addItems(self.serial.portList)#da la lista de los puertos
        #limpiarserial
	def clear_terminal(self):#LIMPIA LA PANTALLA
		self.ui.Terminal.clear()
        #cerrar serial
	def closeEvent(self,e):
		self.serial.disconnect_serial()#desconecta el serial
		#para correr lainterface

if __name__ == '__main__':
	app = QApplication(sys.argv)
	w = MiApp()
	w.show()#muestra la aplicacion
	sys.exit(app.exec_())#corre el programa

