#!/usr/bin/env python3.10

import sys

from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton,
        QLabel, QWidget, QLineEdit,
        QHBoxLayout, QVBoxLayout, QFormLayout,
        QMessageBox)
from PySide6.QtCore import (Qt, Signal, Slot)
from PySide6.QtGui import (QColor)

import cv2
import csv
import os
import numpy as np
import datetime
import time
from PIL import Image, ImageTk

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Face Detector")
        self.layout = QFormLayout()
        self.headingLabel = QLabel("Face Recognition")

        #self.idLabel = QLabel("Enter id")
        self.idLineEdit = QLineEdit()

        #self.nameLabel = QLabel("Enter Name")
        self.nameLineEdit = QLineEdit()

        self.status = QLabel("Saved ImageId : Name : ")

        self.takeImageButton = QPushButton("Take Images")
        self.trainImageButton = QPushButton("Train Images")

        self.layout.addWidget(self.headingLabel)
        self.layout.addRow("Enter id", self.idLineEdit)

        self.layout.addRow("Enter Name", self.nameLineEdit)

        self.layout.addWidget(self.status)

        self.buttons = QWidget()
        self.buttons.layout = QHBoxLayout()
        self.takeImagesButton = QPushButton("Take Images")
        self.trainImagesButton = QPushButton("Train Images")

        self.takeImagesButton.clicked.connect(self.takeImages)
        self.trainImagesButton.clicked.connect(self.trainImages)

        self.buttons.layout.addWidget(self.takeImagesButton)
        self.buttons.layout.addWidget(self.trainImagesButton)

        self.buttons.setLayout(self.buttons.layout)

        self.layout.addWidget(self.buttons)

        self.widget = QWidget()
        self.widget.setLayout(self.layout)

        self.setCentralWidget(self.widget)

        self.show()

    def takeImages(self):
        cam = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        ID = self.idLineEdit.text()
        Name = self.nameLineEdit.text()

        sampleNum = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for(x, y, w, h) in faces:
                cv2.rectangle(img, (x, y),
                        (x+w, y+h), (255, 0, 0), 2)
                sampleNum += 1
                cv2.imwrite("dataset/ " + Name + "." + ID + '.' + str(sampleNum) + '.jpg', 
                        gray[y:y+h, x:x+w])
                print("dataset/ " + Name + "." + ID + '.' + str(sampleNum) + '.jpg')
                #cv2.imshow('Frame', img)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
            if sampleNum > 200:
                break
        cam.release()
        cv2.destroyAllWindows()

        QMessageBox.information(
                self, 'Face Detection',
                f"Saved Image ID : {ID} Name : {Name}")
        pass

    def getImagesAndLabels(self, path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        Ids = []

        for imagePath in imagePaths:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')

            #imageNp = cv2.imread(imagePath)
            #imageNp = cv2.cvtColor(imageNp, cv2.COLOR_BGR2GRAY)

            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(imageNp)
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])
                Ids.append(Id)

        return faceSamples, Ids

    def trainImages(self):

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        global detector
        detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        try:
            global faces, Id
            faces, Id = self.getImagesAndLabels("dataset")
        except Exception as e:
            print(e)

        recognizer.train(faces, np.array(Id))

        try:
            recognizer.save("./model/trained_model2.yml")
        except Exception as e:
            print(e)
        QMessageBox.information(
                self, "Face Detector",
                "Successful trained model with images")
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    app.exec()
