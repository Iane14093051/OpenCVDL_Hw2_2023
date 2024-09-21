import cv2
import math
import numpy as np
import torch
import sys
from PIL import ImageQt, Image
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import pyplot as plt
from torchsummary import summary
import torchvision.models as models
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QLineEdit,
                             QLabel, QPushButton, QMainWindow,  QFileDialog)
from PyQt5.QtWidgets import (QApplication, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QFile, QIODevice
from torchvision import models
from torchvision.transforms import v2
from torchvision import transforms
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic, QtWidgets
import torchvision
import os
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import read_image
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes_2 = ('Cat', 'Dog')


class Gui(QtWidgets.QMainWindow):
    ga_img =[]
    bi_img =[]
    me_img =[]
    ele_structure = [[1,1,1],
               [1,1,1],
               [1,1,1]]
    f_name = []
    num =0 
    def __init__(self,model_path_Vgg, model_path_res, parent=None):
        super(Gui, self).__init__(parent)

        self.image_path = None
        self.model_Vgg = None
        self.model_res = None        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_model_Vgg(model_path_Vgg)
        self.load_model_res(model_path_res)


        self.pix = QPixmap()
        #起点，终点
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        #初始化
        self.initUI()

    def initUI(self):

        # 画布大小为400*400，背景为白色
        self.pix = QPixmap(300, 300)
        self.pix.fill(Qt.black)
        
        self.setWindowTitle("Hw2")
        self.setGeometry(200, 200, 1280, 720)
        self.setFixedSize(1280, 720)

        self.area1 = QtWidgets.QGroupBox("1. Hough Circke Transform", self)
        self.area1.setFont(QtGui.QFont("Arial", 8))
        self.area1.setGeometry(50, 40, 300, 200)

        self.area2 = QtWidgets.QGroupBox("2. Histogram Equalization", self)
        self.area2.setFont(QtGui.QFont("Arial", 8))
        self.area2.setGeometry(50, 340, 300, 300)

        self.area3 = QtWidgets.QGroupBox("3. Morphology Operation", self)
        self.area3.setFont(QtGui.QFont("Arial", 8))
        self.area3.setGeometry(350, 40, 300, 300)

        self.area4 = QtWidgets.QGroupBox("4. MNIST Classifier Using VGG19", self)
        self.area4.setFont(QtGui.QFont("Arial", 8))
        self.area4.setGeometry(350, 340, 500, 350)

        self.area5 = QtWidgets.QGroupBox("5. ResNet50", self)
        self.area5.setFont(QtGui.QFont("Arial", 8))
        self.area5.setGeometry(650, 40, 400, 320)

        color_sep = QPushButton("1.1 Draw contour", self)
        color_sep.setGeometry(60, 60, 160, 50)
        color_sep.clicked.connect(lambda:self.Color(1))

        color_trans = QPushButton("1.2 Count Coins", self)
        color_trans.setGeometry(60, 120, 160, 50)
        color_trans.clicked.connect(lambda:self.Color(2))

        self.count = QLabel('There are _ coins in the image.', self)
        self.count.setGeometry(60, 190, 150, 30)



        Gaussian = QPushButton("2.1 Histogram Equalization", self)
        Gaussian.setGeometry(60, 360, 160, 50)
        Gaussian.clicked.connect(lambda:self.filter(1))

        Load_Image = QPushButton("Load Image", self)
        Load_Image.setGeometry(60, 250, 200, 50)
        Load_Image.clicked.connect(self.Load_Image)

        



        Sobelx = QPushButton("3.1 Closing", self)
        Sobelx.setGeometry(360, 60, 160, 50)
        Sobelx.clicked.connect(lambda:self.mask(1))

        Sobely = QPushButton("3.2 Opening", self)
        Sobely.setGeometry(360, 120, 160, 50)
        Sobely.clicked.connect(lambda:self.mask(2))



 

        Trans = QPushButton("4.1. Show Model Structure", self)
        Trans.setGeometry(360, 360, 160, 50)
        Trans.clicked.connect(lambda:self.Vgg(1))

        Trans = QPushButton("4.2. Show Accuracy an Loss", self)
        Trans.setGeometry(360, 420, 160, 50)
        Trans.clicked.connect(lambda:self.Vgg(2))

        Trans = QPushButton("4.3. Predict", self)
        Trans.setGeometry(360, 480, 160, 50)
        Trans.clicked.connect(lambda:self.Vgg(3))

        Trans = QPushButton("4.4. Reset", self)
        Trans.setGeometry(360, 540, 160, 50)
        Trans.clicked.connect(lambda:self.Vgg(4))

        self.result_label = QLabel(self)
        self.result_label.setGeometry(360, 600, 400, 30) 


        augment = QPushButton("5.1. Show Images", self)
        augment.setGeometry(660, 120, 160, 50)
        augment.clicked.connect(lambda:self.deep(1))


        structure = QPushButton("5.2. Show Model Structure", self)
        structure.setGeometry(660, 180, 160, 50)
        structure.clicked.connect(lambda:self.deep(2))


        Comparison = QPushButton("5.3. Show Comparison", self)
        Comparison.setGeometry(660, 240, 160, 50)
        Comparison.clicked.connect(lambda:self.deep(3))


        Infe = QPushButton("5.4. Inference", self)
        Infe.setGeometry(660, 300, 160, 50)
        Infe.clicked.connect(lambda:self.deep(4))

        Load_img = QPushButton("Load Image", self)
        Load_img.setGeometry(660, 60, 160, 50)
        Load_img.clicked.connect(lambda:self.deep(5))
        
        Predict = QLabel('Predict=', self)
        Predict.setGeometry(830, 60, 160, 50)


        
        self.area6 = QtWidgets.QGroupBox("", self)
        self.area6.setGeometry(900, 60, 128, 128)

        
        self.image_label = QLabel(self)
        self.image_label.setGeometry(900, 60, 128, 128)

        self.result_label_2 = QLabel(self)
        self.result_label_2.setGeometry(840, 80, 160, 50)
        
        

    def Load_Image(self):

        try:
             file_name = QFileDialog.getOpenFileName(self, 'open file', '.')
             self.f_name = file_name[0]

        except:
            pass

    def Color(self,opt):
        if opt ==1:
            self.num =0 
            img = cv2.imread(self.f_name)
            img2 = cv2.imread(self.f_name)
            img3 = cv2.imread(self.f_name)
            h, w, c = img.shape
            for row in range(0,(h)):
                    for col in range(0,(w)):
                         img3[row][col][0]=0
                         img3[row][col][1]=0
                         img3[row][col][2]=0

            
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,20,
            param1=50,param2=30,minRadius=20,maxRadius=40)
            circles = np.uint16(np.around(circles))
            
            for i in circles[0,:]:
             # draw the outer circle
               cv2.circle(img2,(i[0],i[1]),i[2],(0,255,0),2)
             # draw the center of the circle
               cv2.circle(img3,(i[0],i[1]),1,(255,255,255),2)
               self.num+=1
            
            
            cv2.imshow('Img_src',img)       
            cv2.imshow('Img_process',img2)
            cv2.imshow('Circle_center',img3)  
            cv2.waitKey(0)
            #cv2.destoryAllWindows()

        elif opt ==2:
            self.count.setText(f'There are {self.num} coins in the image.')
            self.num=0
            #cv2.destoryAllWindows()
     
        
    def filter(self,opt):
    
        if opt==1:
            img_src = cv2.imread(self.f_name)
            img_src = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)        
            img_dst = cv2.equalizeHist(img_src)
            

            hist_src = cv2.calcHist([img_src], [0], None, [256], [0, 256])
            hist_dst = cv2.calcHist([img_dst], [0], None, [256], [0, 256])

            #显示图像
            fig,ax = plt.subplots(2,3)
            ax[1,0].set_title('Historgram of original')
            ax[1,0].bar(range(256), hist_src.ravel())
            ax[1,0].set_xlabel('Gray scale')
            ax[1,0].set_ylabel('Frequency')
            ax[0,1].set_title('Equalized with OpenCV') 
            ax[0,1].imshow(cv2.cvtColor(img_dst,cv2.COLOR_BGR2RGB),'gray')
            
            
            ax[1,1].set_title('Historgram of Equalized(OpenCV)')
            ax[1,1].bar(range(256), hist_dst.ravel())
            ax[1,1].set_xlabel('Gray scale')
            ax[1,1].set_ylabel('Frequency')        
            ax[0,0].set_title('Original Image')
            ax[0,0].imshow(cv2.cvtColor( img_src, cv2.COLOR_BGR2RGB),'gray')


            
            # Load the image
        
            original_array = np.array(img_src)

            # Step 1: Calculate histogram using numpy.histogram()
            hist, bins = np.histogram(original_array.flatten(), bins=256, range=[0, 256])

            # Step 2: Calculate PDF from normalized histogram
            pdf = hist / np.sum(hist)

            # Step 3: Calculate CDF by cumulatively summing PDF
            cdf = np.cumsum(pdf)

            # Step 4: Create lookup table based on rounded CDF values
            lookup_table = np.round(cdf * 255).astype('uint8')

            # Step 5: Apply lookup table to original image
            equalized_array = lookup_table[original_array]

            # Step 6: Create new equalized image
            equalized_hist, _ = np.histogram(equalized_array.flatten(), bins=256, range=[0, 256])
            

            ax[1,2].set_title('Historgram of Equalized(Manual)')
            ax[1,2].bar(range(256), equalized_hist.ravel())
            ax[1,2].set_xlabel('Gray scale')
            ax[1,2].set_ylabel('Frequency')            
            ax[0,2].set_title('Equalized Manually')
            ax[0,2].imshow(equalized_array, cmap='gray')        


            plt.show()  
            #cv2.destoryAllWindows()

        

        
    def mask(self,opt):
        if opt==1:
            img = cv2.imread(self.f_name)

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0))
            ret2 = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0))
            ret3 = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0))        
            h, w= ret.shape

            


            for row in range(0,(h)):
                    for col in range(0,(w)):
                         if(ret[row][col]<=127):
                             ret[row][col]=0
                         else:
                             ret[row][col]=255
                             
            dst = np.zeros((h-2,w-2), dtype=np.int8)       
            ans=-1000000                     
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         if ans<ret[row-1][col-1]:
                            ans = ret[row-1][col-1]
                         if(ans<ret[row-1][col]):
                            ans = ret[row-1][col]
                         if(ans<ret[row-1][col+1]):
                            ans = ret[row-1][col+1]
                         if(ans<ret[row][col-1]):
                            ans = ret[row][col-1]
                         if(ans<ret[row][col]):
                            ans = ret[row][col]
                         if(ans<ret[row][col+1]):
                            ans = ret[row][col+1]
                         if(ans<ret[row+1][col-1]):
                            ans = ret[row+1][col-1]
                         if(ans<ret[row+1][col]):
                            ans = ret[row+1][col]
                         if(ans<ret[row+1][col+1]):
                            ans = ret[row+1][col+1]
                         ret2[row][col] =ans
                         ans=-1000000


            ans=1000000                     
            h, w= dst.shape
            dst2 = np.zeros((h-2,w-2), dtype=np.int8)
            
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         if(ans>ret2[row-1][col-1]):
                            ans = ret2[row-1][col-1]
                         if(ans>ret2[row-1][col]):
                            ans = ret2[row-1][col]
                         if(ans>ret2[row-1][col+1]):
                            ans = ret2[row-1][col+1]
                         if(ans>ret2[row][col-1]):
                            ans = ret2[row][col-1]
                         if(ans>ret2[row][col]):
                            ans = ret2[row][col]
                         if(ans>ret2[row][col+1]):
                            ans = ret2[row][col+1]
                         if(ans>ret2[row+1][col-1]):
                            ans = ret2[row+1][col-1]
                         if(ans>ret2[row+1][col]):
                            ans = ret2[row+1][col]
                         if(ans>ret2[row+1][col+1]):
                            ans = ret2[row+1][col+1]
                         ret3[row][col] =ans                 
                         ans=1000000

                   
            cv2.imshow("I1",ret3)
            cv2.waitKey(0)
            #cv2.destoryAllWindows()
        elif opt==2:
            img = cv2.imread(self.f_name)

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0))
            ret2 = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0))
            ret3 = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0))        
            h, w= ret.shape

            


            for row in range(0,(h)):
                    for col in range(0,(w)):
                         if(ret[row][col]<=127):
                             ret[row][col]=0
                         else:
                             ret[row][col]=255
                             


            ans=1000000                     
            
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         if(ans>ret[row-1][col-1]):
                            ans = ret[row-1][col-1]
                         if(ans>ret[row-1][col]):
                            ans = ret[row-1][col]
                         if(ans>ret[row-1][col+1]):
                            ans = ret[row-1][col+1]
                         if(ans>ret[row][col-1]):
                            ans = ret[row][col-1]
                         if(ans>ret[row][col]):
                            ans = ret[row][col]
                         if(ans>ret[row][col+1]):
                            ans = ret[row][col+1]
                         if(ans>ret[row+1][col-1]):
                            ans = ret[row+1][col-1]
                         if(ans>ret[row+1][col]):
                            ans = ret[row+1][col]
                         if(ans>ret[row+1][col+1]):
                            ans = ret[row+1][col+1]
                         ret2[row][col] =ans                 
                         ans=1000000
          
            ans=-1000000                     
            for row in range(1,(h-2)):
                    for col in range(1,(w-2)):
                         if ans<ret2[row-1][col-1]:
                            ans = ret2[row-1][col-1]
                         if(ans<ret2[row-1][col]):
                            ans = ret2[row-1][col]
                         if(ans<ret2[row-1][col+1]):
                            ans = ret2[row-1][col+1]
                         if(ans<ret2[row][col-1]):
                            ans = ret2[row][col-1]
                         if(ans<ret2[row][col]):
                            ans = ret2[row][col]
                         if(ans<ret2[row][col+1]):
                            ans = ret2[row][col+1]
                         if(ans<ret2[row+1][col-1]):
                            ans = ret2[row+1][col-1]
                         if(ans<ret2[row+1][col]):
                            ans = ret2[row+1][col]
                         if(ans<ret2[row+1][col+1]):
                            ans = ret2[row+1][col+1]
                         ret3[row][col] =ans
                         ans=-1000000
                   
            cv2.imshow("I1",ret3)
            cv2.waitKey(0)
            #cv2.destoryAllWindows()

    def paintEvent(self, event):
        pp = QPainter(self.pix)
        # 根据鼠标指针前后两个位置绘制直线
        pen = QPen(QColor(Qt.white))
        pen.setWidth(5)  
        pp.setPen(pen) # Set pen color to white


        pp.drawLine(self.lastPoint - QPoint(550, 360), self.endPoint - QPoint(550, 360))  # 转换坐标
        # 让前一个坐标值等于后一个坐标值，
        # 这样就能实现画出连续的线
        self.lastPoint = self.endPoint
        painter = QPainter(self)
        #绘制画布到窗口指定位置处
        painter.drawPixmap(550, 360, self.pix)
      
    def mousePressEvent(self, event):
    # 鼠标左键按下
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.endPoint = self.lastPoint

    def mouseMoveEvent(self, event):
    # 鼠标左键按下的同时移动鼠标
        if event.buttons() and Qt.LeftButton:
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()

    def mouseReleaseEvent(self, event):
    # 鼠标左键释放
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()
    def QPixmapToArray(self):
        ## Get the size of the current pixmap
        size = self.pix.size()
        h = size.width()
        w = size.height()


        ## Get the QImage Item and convert it to a byte string
        qimg = self.pix.toImage()
        b = qimg.bits()
        # sip.voidptr must know size to support python buffer interface
        b.setsize(h * w * 4)
        arr = np.frombuffer(b, np.uint8).reshape((h, w, 4))

        return arr 
            
    def Vgg(self,opt):

        if opt==1:
            device = torch.device("cuda")
            vgg19_bn = models.vgg19_bn(num_classes=10)
            vgg19_bn.to(device)
            summary(vgg19_bn, (3, 32, 32))
            cv2.waitKey(0)     
        elif opt ==2:
            img = cv2.imread('training_curve_Vgg_lee.png')
            cv2.imshow("I1",img)
            cv2.waitKey(0)
        elif opt ==3:
            class_names=['0','1','2','3','4','5','6','7','8','9']
            
            if getattr(self,'pix',None) is None:
                QMessageBox.warning(self, 'Warning','Please load image first!')
                return
            size = self.pix.size()
            h = size.width()
            w = size.height()
            
            #channels_count = 4
            #pixmap = self.pix

            #self.pix.pixmap().save('demo.png','PNG')

            
            #result = pixmap.copy()  # 複製整個 pixmap
            #Q_image = QtGui.QPixmap.toImage(result)
            #grayscale = Q_image.convertToFormat(QtGui.QImage.Format_Grayscale8)
            #s = image.bits().asstring(w * h * channels_count)
            #arr = np.fromstring(s, dtype=np.uint8).reshape((h, w, channels_count))
            # 假設你的 pixmap 變數為 pix
            pix = self.pix

            # 轉換 QPixmap 為 QImage
            image = pix.toImage()

            # 將 QImage 轉換為 NumPy 數組
            size = image.size()
            width = size.width()
            height = size.height()
            channels_count = 4
            s = image.bits().asstring(width * height * channels_count)
            arr = np.fromstring(s, dtype=np.uint8).reshape((height, width, channels_count))

            # 將 NumPy 數組轉換為 Pillow 的 Image 物件
            image_pil = Image.fromarray(arr)
            gray_image_pil = image_pil.convert('L')
            if self.model_res is not None:
                try:
                    transform = v2.Compose([
                                v2.Resize((32, 32)),
                                v2.ToTensor(),
                                v2.Normalize((0.1307,), (0.3081,)),
                                v2.Lambda(lambda x: x.repeat(3, 1, 1)),  
                    ])
                    

                    #image = Image.open('demo.png')
                    transformed_image = transform(gray_image_pil)
                    #transformed_image = transform(image)
                    transformed_image = transformed_image.unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        output = self.model_Vgg(transformed_image)
                    
                    _, predicted = torch.max(output, 1)
                    
                    self.result_label.setText(f"Predicted Class: {class_names[predicted.item()]}")

                    # Plot probability distribution using a histogram
                    probabilities = torch.softmax(output, dim=1).cpu().numpy()
                    plt.bar(class_names, probabilities[0])

                    plt.xlabel("Class")
                    plt.ylabel("Probability")
                    plt.title("Probability Distribution")
                    plt.show()
                    
                except Exception as e:
                    print(f"Error during inference: {e}")
        elif opt ==4:
            self.pix.fill(Qt.black)
            self.lastPoint = QPoint()
            self.endPoint = QPoint()
            self.update()
            
    def deep(self,opt):
        if opt ==1:
            dog_inference_dir = "../Hw2_E14093051_李彥勳_V1/inference_dataset/Dog"
            cat_inference_dir = "../Hw2_E14093051_李彥勳_V1/inference_dataset/Cat"
            transform = v2.Compose([
                                v2.Resize((224, 224))
                    ])
            # Randomly select one dog and one cat image
            dog_image_path = os.path.join(dog_inference_dir, np.random.choice(os.listdir(dog_inference_dir), 1)[0])
            cat_image_path = os.path.join(cat_inference_dir, np.random.choice(os.listdir(cat_inference_dir), 1)[0])

            # Create a 1x2 subplot layout
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            # Display the dog image
            dog_pic = Image.open(dog_image_path)
            
            dog_pic = transform(dog_pic)

            ax[0].imshow(dog_pic)
            ax[0].set_title('Dog')
            ax[0].set_axis_off()

            # Display the cat image
            cat_pic = Image.open(cat_image_path)
            cat_pic = transform(cat_pic)
            ax[1].imshow(cat_pic)
            ax[1].set_title('Cat')
            ax[1].set_axis_off()

            plt.show()
            cv2.waitKey(0)
        elif opt ==2:
            device = torch.device("cuda")
            resnet50_model = models.resnet50(pretrained=True)

            in_features = resnet50_model.fc.in_features
            resnet50_model.fc = nn.Linear(in_features, 1)
            resnet50_model.fc = nn.Sequential(resnet50_model.fc, nn.Sigmoid())
            resnet50_model.to(device)    
            summary(resnet50_model, (3, 224, 224))
            cv2.waitKey(0)
        elif opt ==3:
            img = cv2.imread('compare_lee.png')
            cv2.imshow("I1",img)
            cv2.waitKey(0)
            
        elif opt ==4:
            if hasattr(self, 'image_path') and self.model_res is not None:
                try:
                    transform = v2.Compose([
                                v2.ToTensor(),
                                v2.Resize((224, 224))
                    ])
                    
                    image = Image.open(self.image_path)
                    transformed_image = transform(image)

                    transformed_image = transformed_image.unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        output = self.model_res(transformed_image)
                    
                    x, predicted = torch.max(output, 1)

                    if (x<0.5):
                        self.result_label_2.setText(f"Cat")
                    else:
                        self.result_label_2.setText(f"Dog")

                    # Plot probability distribution using a histogram
                    
                except Exception as e:
                    print(f"Error during inference: {e}")

            
                    
        elif opt ==5:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff);;All Files (*)", options=options)

            if file_name:
                self.image_path = file_name
                pixmap = QPixmap(file_name)
                pixmap = pixmap.scaled(128, 128)
                self.image_label.setGeometry(900, 60,128,128)
                self.image_label.setPixmap(pixmap)

    def load_model_Vgg(self, model_path_Vgg):
        try:
            self.model_Vgg = models.vgg19_bn(num_classes=10)
            self.model_Vgg.load_state_dict(torch.load(model_path_Vgg, map_location=self.device))
            self.model_Vgg.to(self.device)
            self.model_Vgg.eval()
        except Exception as e:
            print(f"Error loading the model: {e}")
    def load_model_res(self, model_path_res):
        try:
            self.model_res = models.resnet50(pretrained=True)
            in_features = self.model_res.fc.in_features
            self.model_res.fc = nn.Linear(in_features, 1)
            self.model_res.fc = nn.Sequential(self.model_res.fc, nn.Sigmoid())           
            self.model_res.load_state_dict(torch.load(model_path_res, map_location=self.device))

            self.model_res.to(self.device) 
            self.model_res.eval()         
        except Exception as e:
            print(f"Error loading the model: {e}")
        

    
def main():
    app = QApplication(sys.argv)
    model_path_Vgg = "best_model_Vgg_lee.pth"  
    model_path_res = "best_model_res_1_lee.pth"   
    root_dir = "../dataset/inference_dataset"
    mainWindow = Gui(model_path_Vgg,model_path_res)
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
