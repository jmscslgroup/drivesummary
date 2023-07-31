# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'visionFinalPROJECT.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1024, 768)
        MainWindow.setStyleSheet("*{\n"
"border: none;\n"
"background-color: transparent;\n"
"background: transparent;\n"
"padding: 0;\n"
"margin: 0;\n"
"color: #fff;\n"
"}\n"
"#centralwidget, #menubar, #statusbar, #homebutton, #stackedwidget, #tabwidget {\n"
"    background-color: #09111c;\n"
"}\n"
"QTabBar{\n"
"    color: #141921;\n"
"    background-color: #141921;\n"
"    font-size: 15px;\n"
"    font-weight: bold;\n"
"}\n"
"\n"
"QLabel, #graphwidget, #acceltab{\n"
"\n"
"    background-color: #141921;\n"
"    border-radius: 4px;\n"
"    padding: 2px;\n"
"    font-size: 15px;\n"
"    font-weight: bold;\n"
"}\n"
"\n"
"#bottomwidget, #mainwidget{\n"
"    background-color:rgb(35, 47, 64);\n"
"}\n"
"\n"
"QProgressBar {\n"
"    border: 2px solid rgb(35, 47, 64);\n"
"    border-radius: 5px;\n"
"    text-align: center;\n"
"    background-color: rgba(33, 37, 43, 180);\n"
"    color: black;\n"
"}\n"
"\n"
"QProgressBar::chunk {\n"
"    background-color: #1e543f;\n"
"    margin: 3px;\n"
"    border-radius: 5px;\n"
"}\n"
"#detailbutton, #homebutton{\n"
"    padding: 5px 10px;\n"
"    border-bottom-left-radius: 10px;\n"
"    border-bottom-right-radius:10px;\n"
"}\n"
"\n"
"#homebutton{\n"
"    border-bottom: 5px solid #1e543f;\n"
"\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.mainwidget = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mainwidget.sizePolicy().hasHeightForWidth())
        self.mainwidget.setSizePolicy(sizePolicy)
        self.mainwidget.setStyleSheet("")
        self.mainwidget.setObjectName("mainwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.mainwidget)
        self.horizontalLayout_2.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.stackedwidget = QtWidgets.QStackedWidget(self.mainwidget)
        self.stackedwidget.setObjectName("stackedwidget")
        self.hubPage = QtWidgets.QWidget()
        self.hubPage.setObjectName("hubPage")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.hubPage)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.leftwidget = QtWidgets.QWidget(self.hubPage)
        self.leftwidget.setObjectName("leftwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.leftwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setSpacing(20)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.ridetimelabel = QtWidgets.QLabel(self.leftwidget)
        self.ridetimelabel.setMinimumSize(QtCore.QSize(200, 80))
        self.ridetimelabel.setMaximumSize(QtCore.QSize(200, 80))
        self.ridetimelabel.setTextFormat(QtCore.Qt.AutoText)
        self.ridetimelabel.setScaledContents(True)
        self.ridetimelabel.setAlignment(QtCore.Qt.AlignCenter)
        self.ridetimelabel.setWordWrap(True)
        self.ridetimelabel.setObjectName("ridetimelabel")
        self.verticalLayout_4.addWidget(self.ridetimelabel, 0, QtCore.Qt.AlignHCenter)
        self.ridedistancelabel = QtWidgets.QLabel(self.leftwidget)
        self.ridedistancelabel.setMinimumSize(QtCore.QSize(280, 120))
        self.ridedistancelabel.setMaximumSize(QtCore.QSize(280, 120))
        self.ridedistancelabel.setScaledContents(True)
        self.ridedistancelabel.setAlignment(QtCore.Qt.AlignCenter)
        self.ridedistancelabel.setWordWrap(True)
        self.ridedistancelabel.setObjectName("ridedistancelabel")
        self.verticalLayout_4.addWidget(self.ridedistancelabel, 0, QtCore.Qt.AlignHCenter)
        self.ccpercentlabel = QtWidgets.QLabel(self.leftwidget)
        self.ccpercentlabel.setMinimumSize(QtCore.QSize(200, 80))
        self.ccpercentlabel.setMaximumSize(QtCore.QSize(200, 80))
        self.ccpercentlabel.setScaledContents(True)
        self.ccpercentlabel.setWordWrap(True)
        self.ccpercentlabel.setObjectName("ccpercentlabel")
        self.verticalLayout_4.addWidget(self.ccpercentlabel)
        self.cctimelabel = QtWidgets.QLabel(self.leftwidget)
        self.cctimelabel.setMinimumSize(QtCore.QSize(200, 80))
        self.cctimelabel.setMaximumSize(QtCore.QSize(200, 80))
        self.cctimelabel.setScaledContents(True)
        self.cctimelabel.setWordWrap(True)
        self.cctimelabel.setObjectName("cctimelabel")
        self.verticalLayout_4.addWidget(self.cctimelabel)
        self.verticalLayout_2.addLayout(self.verticalLayout_4)
        self.horizontalLayout_3.addWidget(self.leftwidget)
        self.rightwidget = QtWidgets.QWidget(self.hubPage)
        self.rightwidget.setObjectName("rightwidget")
        self.stopslabel = QtWidgets.QLabel(self.rightwidget)
        self.stopslabel.setGeometry(QtCore.QRect(170, 50, 140, 70))
        self.stopslabel.setMinimumSize(QtCore.QSize(140, 70))
        self.stopslabel.setMaximumSize(QtCore.QSize(140, 70))
        self.stopslabel.setScaledContents(True)
        self.stopslabel.setAlignment(QtCore.Qt.AlignCenter)
        self.stopslabel.setWordWrap(True)
        self.stopslabel.setObjectName("stopslabel")
        self.graphwidget = QtWidgets.QWidget(self.rightwidget)
        self.graphwidget.setGeometry(QtCore.QRect(10, 150, 470, 450))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphwidget.sizePolicy().hasHeightForWidth())
        self.graphwidget.setSizePolicy(sizePolicy)
        self.graphwidget.setMinimumSize(QtCore.QSize(470, 450))
        self.graphwidget.setMaximumSize(QtCore.QSize(470, 450))
        self.graphwidget.setObjectName("graphwidget")
        self.horizontalLayout_3.addWidget(self.rightwidget)
        self.stackedwidget.addWidget(self.hubPage)
        self.infoPage = QtWidgets.QWidget()
        self.infoPage.setObjectName("infoPage")
        self.stackedwidget.addWidget(self.infoPage)
        self.horizontalLayout_2.addWidget(self.stackedwidget)
        self.verticalLayout.addWidget(self.mainwidget)
        self.bottomwidget = QtWidgets.QWidget(self.centralwidget)
        self.bottomwidget.setStyleSheet("")
        self.bottomwidget.setObjectName("bottomwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.bottomwidget)
        self.horizontalLayout.setContentsMargins(11, 0, 11, 11)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.homebutton = QtWidgets.QPushButton(self.bottomwidget)
        self.homebutton.setMinimumSize(QtCore.QSize(50, 50))
        self.homebutton.setMaximumSize(QtCore.QSize(50, 50))
        self.homebutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.homebutton.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../../.designer/backup/images/25694.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon.addPixmap(QtGui.QPixmap("../../../.designer/backup/images/25694.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.homebutton.setIcon(icon)
        self.homebutton.setIconSize(QtCore.QSize(30, 30))
        self.homebutton.setObjectName("homebutton")
        self.horizontalLayout.addWidget(self.homebutton)
        self.detailbutton = QtWidgets.QPushButton(self.bottomwidget)
        self.detailbutton.setMinimumSize(QtCore.QSize(50, 50))
        self.detailbutton.setMaximumSize(QtCore.QSize(50, 50))
        self.detailbutton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.detailbutton.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../../../.designer/backup/images/221407.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon1.addPixmap(QtGui.QPixmap("../../../.designer/backup/images/221407.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.detailbutton.setIcon(icon1)
        self.detailbutton.setIconSize(QtCore.QSize(30, 30))
        self.detailbutton.setObjectName("detailbutton")
        self.horizontalLayout.addWidget(self.detailbutton)
        self.verticalLayout.addWidget(self.bottomwidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.menuFile.addAction(self.actionOpen)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.ridetimelabel.setText(_translate("MainWindow", "Ride Time"))
        self.ridedistancelabel.setText(_translate("MainWindow", "Ride Miles"))
        self.ccpercentlabel.setText(_translate("MainWindow", "Cruise Control Time"))
        self.cctimelabel.setText(_translate("MainWindow", "Cruise Control Percentage"))
        self.stopslabel.setText(_translate("MainWindow", "Stops"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())