# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'prueba1.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(764, 503)
        MainWindow.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.terminar = QtWidgets.QPushButton(self.centralwidget)
        self.terminar.setGeometry(QtCore.QRect(50, 630, 867, 28))
        self.terminar.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.terminar.setObjectName("terminar")
        self.encender = QtWidgets.QPushButton(self.centralwidget)
        self.encender.setGeometry(QtCore.QRect(10, 150, 261, 41))
        self.encender.setStyleSheet("background-color: rgb(170, 85, 0);")
        self.encender.setCheckable(False)
        self.encender.setChecked(False)
        self.encender.setObjectName("encender")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 50, 261, 41))
        self.label.setStyleSheet("background-color: rgb(0, 194, 194);")
        self.label.setObjectName("label")
        self.camara = QtWidgets.QComboBox(self.centralwidget)
        self.camara.setGeometry(QtCore.QRect(10, 100, 261, 41))
        self.camara.setStyleSheet("\n"
"background-color: rgb(255, 255, 255);")
        self.camara.setObjectName("camara")
        self.camara.addItem("")
        self.camara.addItem("")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 200, 261, 231))
        self.label_2.setStyleSheet("\n"
"background-color: rgb(0, 85, 255);")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, -10, 741, 51))
        self.label_3.setStyleSheet("\n"
"background-color: rgb(170, 0, 127);")
        self.label_3.setObjectName("label_3")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(280, 50, 471, 391))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton.setStyleSheet("\n"
"background-color: rgb(202, 128, 255);")
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.resultado = QtWidgets.QTextEdit(self.layoutWidget)
        self.resultado.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"")
        self.resultado.setObjectName("resultado")
        self.verticalLayout.addWidget(self.resultado)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 764, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(self.resultado.clear)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.terminar.setText(_translate("MainWindow", "TERMINAR"))
        self.encender.setText(_translate("MainWindow", "DETECTAR"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:14pt; color:#ffffff;\">ELEGIR CAMARA</span></p></body></html>"))
        self.camara.setItemText(0, _translate("MainWindow", "0"))
        self.camara.setItemText(1, _translate("MainWindow", "1"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:26pt; font-weight:600; font-style:italic; color:#ffffff;\">AG</span></p><p align=\"center\"><span style=\" font-size:26pt; font-weight:600; font-style:italic; color:#ffffff;\">Electrónica</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt; color:#ffffff;\">DETECTOR DE COMPONENTES ELECTRONICOS </span></p></body></html>"))
        self.pushButton.setText(_translate("MainWindow", "CLEAR"))

