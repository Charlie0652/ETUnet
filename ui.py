# -*- coding: utf-8 -*-
# 应该在界面启动的时候就将模型加载出来，设置tmp的目录来放中间的处理结果
import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import torch
import os.path as osp
# from TBUnet import ConvUNeXt
from ETUNet import ETUNET
# from Models.dnetv25v3 import TNet
import numpy as np
# from Data import dataloaders
from PIL import Image
from torchvision import transforms

# 窗口主类
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')

transform = transforms.Compose([
    transforms.ToTensor()
])

class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('黑色素瘤皮肤镜图像分割系统')
        self.resize(1024, 800)
        self.setWindowIcon(QIcon("images/UI/lufei.png"))
        # 图片读取进程
        self.output_size = 512
        self.img2predict = ""
        # # 初始化视频读取线程
        self.origin_shape = (512, 512)
        # 加载网络，图片单通道，分类为1。
        checkpoint_path = "E:/lijiawei/ETUnet-main/data/ETUnet-checkpoint.pth"
        # checkpoint_path = "E:/lijiawei/ETUnet-main/data/Isic2017-checkpoint.pth"
        net = ETUNET()
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))

        # 测试模式
        net.eval()
        self.model = net.cuda()
        self.initUI()

    '''
    ***界面初始化***
    '''

    def initUI(self):
        # 图片检测子界面
        font_title = QFont('宋体', 22)
        font_main = QFont('宋体', 18)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("黑色素瘤皮肤镜图像分割功能")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始分割")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:black}"
                                    "QPushButton:hover{background-color: rgb(23,216,230);}"
                                    "QPushButton{background-color:rgb(51,102,204)}"
                                    "QPushButton{border:5px}"
                                    "QPushButton{border-radius:10px}"
                                    "QPushButton{padding:10px 10px}"
                                    "QPushButton{margin:10px 300px 10px 300px}")
        det_img_button.setStyleSheet("QPushButton{color:black}"
                                     "QPushButton:hover{background-color: rgb(23,216,230);}"
                                     "QPushButton{background-color:rgb(51,102,204)}"
                                     "QPushButton{border:5px}"
                                     "QPushButton{border-radius:10px}"
                                     "QPushButton{padding:10px 10px}"
                                     "QPushButton{margin:10px 300px 10px 300px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # todo 关于界面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用黑色素瘤皮肤镜图像分割系统\n')  # todo 修改欢迎词语
        about_title.setFont(QFont('宋体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/UI/qq.png'))
        about_img.setAlignment(Qt.AlignCenter)

        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        #about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        pre_widget = QWidget()
        pre_layout = QVBoxLayout()
        pre_title = QLabel('欢迎使用黑色素瘤皮肤镜图像分割系统\n')
        pre_title.setFont(QFont('宋体', 18))
        pre_title.setAlignment(Qt.AlignCenter)
        pre_img = QLabel()
        pre_img.setPixmap(QPixmap('images/UI/qq.png'))
        pre_img.setAlignment(Qt.AlignCenter)
        pre_layout.addWidget(pre_title)
        pre_layout.addStretch()
        pre_layout.addWidget(pre_img)
        pre_layout.addStretch()
        pre_widget.setLayout(pre_layout)

        post_widget = QWidget()
        post_layout = QVBoxLayout()
        post_title = QLabel('欢迎使用黑色素瘤皮肤镜图像分割系统\n')
        post_title.setFont(QFont('宋体', 18))
        post_title.setAlignment(Qt.AlignCenter)
        post_img = QLabel()
        post_img.setPixmap(QPixmap('images/UI/qq.png'))
        post_img.setAlignment(Qt.AlignCenter)
        post_layout.addWidget(post_title)
        post_layout.addStretch()
        post_layout.addWidget(post_img)
        post_layout.addStretch()
        post_widget.setLayout(post_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '黑色素瘤分割')
        # self.addTab(pre_widget, '预处理')
        # self.addTab(post_widget, '后处理')
        self.addTab(about_widget, '帮助')
        self.setTabIcon(0, QIcon('images/UI/lufei.png'))
        self.setTabIcon(1, QIcon('images/UI/lufei.png'))
        # self.setTabIcon(2, QIcon('images/UI/lufei.png'))
        # self.setTabIcon(3, QIcon('images/UI/lufei.png'))

    '''
    ***上传图片***
    '''

    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')

        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("./uires", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            # resize_scale = self.output_size / im0.shape[0]
            # im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)

            im0 = cv2.resize(im0, (512, 512))
            cv2.imwrite("./uires/tmp_upload.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            print(fileName)
            self.img2predict = "E:/lijiawei/ETUnet-main/uires/tmp_upload.jpg"
            self.origin_shape = (im0.shape[1], im0.shape[0])
            self.left_img.setPixmap(QPixmap("./uires/tmp_upload.jpg"))
            # todo 上传图片之后右侧的图片重置，
            # self.right_img.setPixmap(QPixmap("./uires/tmp_upload.jpg"))
            self.right_img.setPixmap(QPixmap(""))

    '''
    ***检测图片***
    '''
    def detect_img(self):
        model = self.model
        output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        img = Image.open(source)
        temp = max(img.size)
        mask = Image.new('RGB', (temp, temp))
        mask.paste(img, (0, 0))
        mask = mask.resize((512, 512))
        img_data = transform(mask).cuda()
        img_data = torch.unsqueeze(img_data, dim=0)
        output= model(img_data)
        predicted_map = np.array(output.cpu().detach().numpy())
        predicted_map = np.squeeze(predicted_map)
        predicted_map = predicted_map > 0
        cv2.imwrite("./uires/single_result.jpg", predicted_map * 255)

        # img = cv2.imread(source)
        # print(img)
        # # 转为灰度图
        # # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img = cv2.resize(img, (352, 352))
        # print(img.shape)
        # # 转为batch为1，通道为1，大小为512*512的数组
        # img = img.reshape(1, img.shape[2], img.shape[0], img.shape[1])
        # # 转为tensor
        # img_tensor = torch.from_numpy(img)
        # # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        # img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # # 预测
        # #pred = model(img_tensor)
        # pred, _ = model(img_tensor)
        # print(pred)
        # # 提取结果
        # # pred = np.array(pred[4].detach().numpy())
        # pred = np.array(pred.cpu().detach().numpy())
        # # 处理结果
        # # pred[pred >= 0.5] = 255
        # # pred[pred < 0.5] = 0
        # pred = np.squeeze(pred)
        # pred = pred > 0
        # print(pred)
        # # # 保存图片
        # cv2.imwrite("./uires/single_result.jpg", pred * 255)
        # 目前的情况来看，应该只是ubuntu下会出问题，但是在windows下是完整的，所以继续
        self.right_img.setPixmap(QPixmap("./uires/single_result.jpg"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
