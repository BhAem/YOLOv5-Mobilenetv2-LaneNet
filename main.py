
import cv2
import numpy as np
from sort import Sort
from detector import Detector

import time
import sys
from PyQt5 import QtGui
from PyQt5.Qt import *
from lanes import _load_model, _frame_process
from clustering import lane_cluster
import lanes

class MainDemo(QWidget):

    def __init__(self):
        super().__init__()
        # 控制、交互消息
        self.isVIDEO = False  # 在运行open_video函数得到路径之前默认无视频
        self.isPICTURE = False  # 是否已经传入图片
        self.video_name = None  # 在运行open_video函数得到路径之前默认为None
        self.initial_img_name = None  # 选择打开的图片路径，打开之前默认为None
        self.result_img_name = None  # 处理之后的图片路径

        self.new_line=None

        # 一些需要传递给counting的数据
        self.line=None
        self.dots=None
        self.original_shape=None
        self.segmentation_mask=None

        # 图片显示区域
        self.img_label = QLabel()  # 放置图片的label
        self.img_label.setText("图 片 显 示 区 域")
        self.img_label.setStyleSheet('font:bold; font-size:80px;border-width: 1px;border-style: solid;'
                                     'color: blue;background-color: rgb(220, 220, 220);')
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setFixedSize(1250, 710)

        # 下方为 具体消息的标题 显示
        message_title_left = QLabel("各车道信息：")  # 各车道信息标题
        message_title_left.setAlignment(Qt.AlignCenter)
        message_title_left.setFixedSize(310,40)
        message_title_left.setStyleSheet('font:bold; font-size:25px;')
        message_title_mid = QLabel(" F P S ：")  # FPS标题
        message_title_mid.setFixedSize(310,80)
        message_title_mid.setAlignment(Qt.AlignCenter)
        message_title_mid.setStyleSheet('font:bold; font-size:25px;')
        self.messageLabelLeft = QLabel()   # 具体 车流量信息 显示
        self.messageLabelCenter = QLabel()  # 具体信息显示中间


        self.messageLabelLeft.setFixedSize(310,400)
        self.messageLabelCenter.setFixedSize(310, 30)

        self.messageLabelLeft.setAlignment(Qt.AlignCenter)  #居中
        self.messageLabelCenter.setAlignment(Qt.AlignCenter)

        self.messageLabelLeft.setStyleSheet('border-width: 1px;border-style: solid;font-size:30px;'
                                            'color: blue;background-color: rgb(220, 220, 220);')
        self.messageLabelCenter.setStyleSheet('font-size:25px;')


        # 窗口整体设计
        self.setWindowTitle("基于人工智能的交通信号灯控制系统")  # 标题
        self.setFixedSize(1630, 850)           # 窗口尺寸
        self.setWindowIcon(QIcon('./data/images/1.png'))  # 程序图标

        # 各个按钮设置与相关事件
        self.open_image_button = QPushButton("打开图片")
        self.open_image_button.clicked.connect(self.open_image)  # 打开图片（获取图片路径）
        self.open_image_button.setFixedSize(160, 40)
        self.open_image_button.setStyleSheet('font-size:24px;')
        self.lanes_button = QPushButton("检测车道")
        self.lanes_button.clicked.connect(self.show_lanes)
        self.lanes_button.setFixedSize(160, 40)
        self.lanes_button.setStyleSheet('font-size:24px;')
        self.open_video_button = QPushButton("打开视频")
        self.open_video_button.clicked.connect(self.open_video)  # 打开视频（获取视频路径）
        self.open_video_button.setFixedSize(160, 40)
        self.open_video_button.setStyleSheet('font-size:24px;')
        self.result_file_button = QPushButton("开始运行")  # 实时显示处理后图片与数据信息
        self.result_file_button.clicked.connect(self.open_result_image)
        self.result_file_button.setFixedSize(160, 40)
        self.result_file_button.setStyleSheet('font-size:24px;')

        # 按钮行水平放置
        line_1 = QWidget(self)
        layout = QHBoxLayout() # 水平布局放置按钮
        layout.addWidget(self.open_image_button)  # 添入open_image_button的按钮
        layout.addWidget(self.lanes_button)  # 同上
        layout.addWidget(self.open_video_button)  # 同上
        layout.addWidget(self.result_file_button)  # 同上
        layout.setAlignment(Qt.AlignCenter)   # 水平放置方式为 靠左AlignLeft 居中Qt.AlignCenter
        line_1.setLayout(layout)    # 应用布局
        line_1.setFixedSize(1200, 50)  # 设置按钮行尺寸

        # 实时信息具体内容
        self.messageLabelLeft.setText("具体信息等待显示")
        self.messageLabelCenter.setText("0")

        # 实时信息标题
        layout = QVBoxLayout()  # 垂直布局放置实时信息标题
        layout.addWidget(message_title_left)
        layout.addWidget(self.messageLabelLeft)
        layout.addWidget(message_title_mid)
        layout.addWidget(self.messageLabelCenter)
        line_3 = QWidget()    # 放置实时信息的容器
        line_3.setLayout(layout)
        line_3.setFixedSize(340,740)

        # 图片label的布局方式
        self.pic_layout = QHBoxLayout()  # 水平布局放置图片
        self.pic_layout.addWidget(self.img_label)  # 添入选择的初始图片
        self.pic_layout.addWidget(line_3)  # 添入车道实时信息
        line_2 = QWidget()
        line_2.setLayout(self.pic_layout)  # 应用布局
        line_2.setFixedSize(1630, 760)


        # 总体是垂直布局的,line_1 按钮行 line_2 内容行 line_3 右侧信息竖栏（Line_3放入line_2）
        layout = QVBoxLayout()  # 将 水平摆放的按钮行 与 图片区域 垂直摆放
        layout.addWidget(line_1)  # 添入 按钮行
        layout.addWidget(line_2)  # 添入 中间水平框
        self.setLayout(layout)  # 将布局应用于整个窗口

    def open_image(self):
        """
        修改self.isPICTURE属性为True
        获得初始图片路径，并将其传给lanes.py中相关函数
        """
        self.isPICTURE = True  # 修改self.isPICTURE属性为True
        self.initial_img_name, _ = QFileDialog.getOpenFileName(self, "选择需要处理的图片", "*.jpg;;*.png;;*.jpeg")  # 获取打开的图片路径
        print(self.initial_img_name)   # 打印图片路径
        initial_img = QtGui.QPixmap(self.initial_img_name)
        self.img_label.setPixmap(initial_img)

        # 运行lanes.py文件中的主函数


    def open_video(self):
        """
        修改self.isVIDEO属性为True
        获取视频路径
        """
        self.isVIDEO = True  # 传入的为视频，修改属性
        self.video_name, _ = QFileDialog.getOpenFileName(self, "选择需要处理的视频", "*.mp4;;*.avi")  # 获取选定的视频路径
        print(self.video_name)  # 打印视频路径

    def open_result_image(self):
        """
        开始运行，在start_counting里面自动产生图片，调用show_pictures
        """
        # 判断是否正确选择了视频与图片
        if self.isPICTURE and self.isVIDEO :
            self.start_counting(self.line, self.dots, self.original_shape, self.segmentation_mask)  # 开始计数
        elif self.isPICTURE :
            print("你还没有选择视频！")
            message = QMessageBox(QMessageBox.Warning, '提示', '你还没有选择视频！')  # 弹窗提示
            message.exec_()
        elif self.isVIDEO :
            print("你还没有选择图片！")
            message = QMessageBox(QMessageBox.Warning, '提示', '你还没有选择图片！')  # 弹窗提示
            message.exec_()
        elif not(self.isPICTURE and self.isVIDEO):
            print("你还没有选择图片与视频！")
            message = QMessageBox(QMessageBox.Warning, '提示', '你还没有选择图片与视频！')  # 弹窗提示
            message.exec_()

    # 显示实时车辆检测图
    def show_pictures(self,image):
        """
        修改label中图片,展示效果图片
        """
        # 根据传入的路径，将图片导入到界面上
        self.result_img_name = image
        result_img = QtGui.QPixmap(self.result_img_name)
        self.img_label.setPixmap(result_img)  # 放置图片

    def show_lanes(self):
        """
        修改label中图片，显示车道线信息
        """
        # 根据传入的路径，将图片导入到界面上
        if self.isPICTURE:
            input_ad = self.initial_img_name
            model_path = './weights/model_1_1650561948_23_5.693099021911621.pkl'
            image_size = tuple([512, 256])
            threshold = .5
            bandwidth = 3

            model = _load_model(model_path)
            model.eval()
            img_frame = cv2.imread(input_ad, cv2.IMREAD_UNCHANGED)
            self.original_shape = (1280, 720)

            frame_height, frame_width = 720, 1280

            embeddings, threshold_mask, img = _frame_process(img_frame, model, image_size, threshold)
            cluster = lane_cluster(bandwidth, img, embeddings.squeeze().data.cpu().numpy(), threshold_mask,
                                   method='Meanshift')
            fitted_image, instance_mask, self.segmentation_mask, lane_idx, labels, unique_label = cluster()

            instance_mask = cv2.cvtColor(instance_mask, cv2.COLOR_RGB2BGR)
            fitted_image = cv2.cvtColor(fitted_image, cv2.COLOR_RGB2BGR)

            image = cv2.resize(instance_mask, self.original_shape, interpolation=cv2.INTER_NEAREST)  # 得到的车道线分割图片
            cv2.imwrite("data/images/lanes.jpg", image)  # 需要呈现出来的图片
            # cv2.imshow("lane", image)
            # cv2.waitKey(0)
            self.segmentation_mask = cv2.resize(self.segmentation_mask, self.original_shape,
                                                interpolation=cv2.INTER_NEAREST)
            # cv2.imshow("lane", self.segmentation_mask)
            # cv2.waitKey(0)

            line_height = int(frame_height * 15 / 24)
            self.line = [(0, line_height), (int(frame_width), line_height)]  # 撞线

            dots1 = []
            for i in range(frame_width):
                if self.segmentation_mask[line_height, i, 0] == 255:
                    dots1.append(i)

            threshold = 50
            temp = []
            self.dots = []
            self.dots.append(0)
            for idx, i in enumerate(dots1):
                if idx == 0:
                    temp.append(i)
                else:
                    if dots1[idx] - dots1[idx - 1] <= threshold:
                        temp.append(i)
                    else:
                        SUM = np.sum(temp)
                        self.dots.append(int(SUM / len(temp)))
                        temp = []
                        temp.append(dots1[idx])
            SUM = np.sum(temp)
            self.dots.append(int(SUM / len(temp)))
            self.dots.append(frame_width)
            self.result_img_name = "data/images/lanes.jpg"
            result_img = QtGui.QPixmap(self.result_img_name)
            self.img_label.setPixmap(result_img)  # 放置图片
        else:
            print("请先选择待检测车道图片！")
            message = QMessageBox(QMessageBox.Warning, '提示', '请先选择待检测车道图片！')  # 弹窗提示
            message.exec_()


    # 显示实时车辆数信息
    def change_details(self,counters,fps):
        """
        用来修改各个车道信息,显示在界面上
        """
        string=""
        for idx, kk in enumerate(counters):
            string += "Road" + str(idx) + ": " + str(kk) + "\n"

        self.messageLabelLeft.setText(string)
        fps = round(fps, 3)
        self.messageLabelCenter.setText(str(fps))


    # 车辆计数
    def start_counting(self,line2, dots2, original_shape, segmentation_mask):
        capture = cv2.VideoCapture(self.video_name)  # 读取视频流，也可以是开启摄像头
        # capture = cv2.VideoCapture(0) # 开启摄像头
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        line = line2  # 撞线
        # deta_y = 5  # 自定义△y
        # y = int(frame_height * 2/3) - deta_y  # 自定义y1
        # deta1 = abs(lane[0][2] - lane[1][2])  # 左半部分的车道宽
        # deta2 = abs(lane[2][2] - lane[3][2])  # 右半部分的车道宽
        dots = dots2  #######
        # 利用相似三角形计算距离撞线较近的点
        cnt = 0
        # for x1, y1, x2, y2 in lane:
        #     cnt += 1
        #     if cnt <= 2:
        #         x3 = ((x2 - x1) / (y2 - y1)) * (y - y1) + x1
        #     elif cnt > 2:
        #         x3 = -(((x1 - x2) / (y1 - y2)) * (y1 - y) - x1)
        #     dots.append(int(x3))

        tracker = Sort()  # 创建跟踪器类对象
        memory = {}  # 用于保存当前帧跟踪成功的跟踪框（建立id和检测框的映射）
        # 视频的宽度和高度，即帧尺寸
        (W, H) = (None, None)
        detector = Detector()  # 创建检测类对象
        # 计数变量
        num_lanes = len(dots2) - 2
        counters = [0 for i in range(num_lanes + 1)]
        fps = 0.0  # 计算帧数
        while True:
            t1 = time.time()
            (grabed, frame) = capture.read()
            if not grabed:
                break
            frame = cv2.resize(frame, original_shape, interpolation=cv2.INTER_NEAREST)
            # cv2.imwrite("data/images/4cloud.png", frame)
            # 绘制车道线
            # for idx in range(1, num_lanes):
            #     cv2.line(frame, (dots[idx], line[0][1]-200), (dots[idx], line[0][1]+200), (255, 0, 0), 10)
            frame = cv2.addWeighted(frame, .9, segmentation_mask, 1, 0)
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            dets = detector.detect(frame)  # 得到检测框

            # print("frame____________________________")
            # for det in dets:
            #     print(det[4])

            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})  # 将整型数据转换为浮点数类型
            dets = np.asarray(dets)  # 将检测框数据转换为ndarray,其数据类型为浮点型

            if np.size(dets) == 0:  # 未检测到对象
                continue
            else:
                tracks = tracker.update(dets)  # 跟踪器更新检测框，即将预测框和检测框进行比对
            boxes = []  # 存储当前帧中跟踪目标成功的跟踪框
            ids = []  # 存储车辆id
            pre = memory.copy()  # 把上一帧保留下来的跟踪成功的跟踪框复制一份
            memory = {}  # 置空
            # 遍历当前帧中跟踪目标成功的跟踪框
            for track in tracks:
                boxes.append([track[0], track[1], track[2], track[3], ])
                ids.append(int(track[4]))
                memory[ids[-1]] = boxes[-1]
            # 使用虚拟线圈进行检测
            if len(boxes) > 0:
                i = 0
                for box in boxes:
                    (x, y) = (int(box[0]), int(box[1]))
                    (w, h) = (int(box[2]), int(box[3]))
                    frame = self.draw_bboxes(frame, tracks, line_thickness=None)
                    if ids[i] in pre:
                        previous_box = pre[ids[i]]
                        (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                        (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                        p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))  # 上一帧中跟踪框的中心点坐标
                        p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))  # 当前帧中跟踪框的中心点坐标
                        # 判断直线p0p1和虚拟线圈进行相交
                        if self.intersect(line[0], line[1], p0, p1):
                            for kk in range(1, num_lanes + 1):
                                if dots[kk - 1] <= p0[0] <= dots[kk]:
                                    counters[kk - 1] += 1
                                elif kk == num_lanes and p0[0] > dots[kk]:
                                    counters[kk] += 1
                            # if y2 > y:
                            #     # 逆向车道的车辆数据
                            #     if p0[0] <= dots[0]:
                            #         counter_up_left += 1
                            #     elif dots[1] >= p0[0] >= dots[0]:
                            #         counter_up_mid += 1
                            #     elif dots[1] + deta1 >= p0[0] >= dots[1]:
                            #         counter_up_right += 1
                            # else:
                            #     # 正向车道的车辆数据
                            #     if dots[2] - deta2 <= p0[0] <= dots[2]:
                            #         counter_down_left += 1
                            #     elif dots[2] <= p0[0] <= dots[3]:
                            #         counter_down_mid += 1
                            #     elif dots[3] <= p0[0] <= dots[3] + deta2:
                            #         counter_down_right += 1
                    i += 1
            fps = (fps + (1. / (time.time() - t1))) / 2  # 计算平均fps
            # 使用opencv进行视频的绘制
            cv2.line(frame, line[0], line[1], (70, 219, 255), 3)
            text_draw1 = ""
            # for idx, kk in enumerate(counters):
            #
            #     text_draw1 += "Road" + str(idx) + ": " + str(kk) + " "
            # text_draw1 = 'Left1: ' + str(counter_up_left) + ' Left2: ' + str(counter_up_mid) + ' Left3: ' + str(counter_up_right)
            # text_draw2 = 'FPS: %.2f' % fps

            # # text_draw3 = 'Right1: ' + str(counter_down_left) + ' Right2: ' + str(counter_down_mid) + ' Right3: ' + str(counter_down_right)
            # font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
            # draw_text_postion1 = (int(frame_width * 0.01), int(frame_height * 0.05))
            # draw_text_postion2 = (int(frame_width * 0.4), int(frame_height * 0.1))
            # # draw_text_postion3 = (int(frame_width * 0.57), int(frame_height * 0.05))
            # handled_image_frame = cv2.putText(img=frame, text=text_draw1, org=draw_text_postion1,
            #                                   fontFace=font_draw_number, fontScale=1, color=(60, 179, 113), thickness=2)
            # handled_image_frame = cv2.putText(img=frame, text=text_draw2, org=draw_text_postion2,
            #                                   fontFace=font_draw_number, fontScale=1, color=(60, 179, 113), thickness=2)
            # # handled_image_frame = cv2.putText(img=frame, text=text_draw3, org=draw_text_postion3,
            # #                                   fontFace=font_draw_number, fontScale=1, color=(60, 179, 113), thickness=2)

            # 实时信息
            cv2.imwrite("data/images/result.jpg", frame)
            self.show_pictures("data/images/result.jpg") # 车道图片
            self.change_details(counters, fps)   # 车流量信息等
            # cv2.imshow('demo', handled_image_frame)
            cv2.waitKey(1)
        print(fps)
        capture.release()
        cv2.destroyAllWindows()

    # 计算叉乘
    def intersect(self,line1, line2, p0, p1):
        cross_product1 = (p0[1] - line1[1]) * (line2[0] - line1[0]) > (line2[1] - line1[1]) * (p0[0] - line1[0])
        cross_product2 = (p1[1] - line1[1]) * (line2[0] - line1[0]) > (line2[1] - line1[1]) * (p1[0] - line1[0])
        return cross_product1 != cross_product2

    # 画图
    def draw_bboxes(self,image, bboxes, line_thickness):
        line_thickness = line_thickness or round(0.001 * (image.shape[0] + image.shape[1]) * 0.2) + 1
        for (x1, y1, x2, y2, pos_id, cls_id) in bboxes:
            type = "car"
            color = (0, 151, 255)
            c1, c2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
            font_thickness = max(line_thickness - 1, 1)
            t_size = cv2.getTextSize(str(cls_id), 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
            c2 = c1[0] + t_size[0] + 7, c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(image, type, (c1[0], c1[1] - 2), 0, line_thickness / 3,
                        [0, 0, 0], thickness=font_thickness, lineType=cv2.LINE_AA)
        return image


if __name__ == '__main__':
    app = QApplication(sys.argv)
    run = MainDemo()
    run.show()
    app.exec_()