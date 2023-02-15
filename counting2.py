import time
import cv2
import numpy as np
from sort import Sort
from detector import Detector

def start_counting(line2, dots2, original_shape, segmentation_mask):
    capture = cv2.VideoCapture('data/images/input4.mp4')  # 读取视频流，也可以是开启摄像头
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
    num_lanes = len(dots2)-2
    counters = [0 for i in range(num_lanes+1)]
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
                frame = draw_bboxes(frame, tracks, line_thickness=None)
                if ids[i] in pre:
                    previous_box = pre[ids[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))  # 上一帧中跟踪框的中心点坐标
                    p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))  # 当前帧中跟踪框的中心点坐标
                    # 判断直线p0p1和虚拟线圈进行相交
                    if intersect(line[0], line[1], p0, p1):
                        for kk in range(1, num_lanes+1):
                            if dots[kk-1] <= p0[0] <= dots[kk]:
                                counters[kk-1] += 1
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
        for idx, kk in enumerate(counters):
            text_draw1 += "Road" + str(idx) + ": " + str(kk) + " "
        # text_draw1 = 'Left1: ' + str(counter_up_left) + ' Left2: ' + str(counter_up_mid) + ' Left3: ' + str(counter_up_right)
        text_draw2 = 'FPS: %.2f' % fps
        # text_draw3 = 'Right1: ' + str(counter_down_left) + ' Right2: ' + str(counter_down_mid) + ' Right3: ' + str(counter_down_right)
        font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
        draw_text_postion1 = (int(frame_width * 0.01), int(frame_height * 0.05))
        draw_text_postion2 = (int(frame_width * 0.4), int(frame_height * 0.1))
        # draw_text_postion3 = (int(frame_width * 0.57), int(frame_height * 0.05))
        handled_image_frame = cv2.putText(img=frame, text=text_draw1, org=draw_text_postion1,
                                          fontFace=font_draw_number, fontScale=1, color=(60, 179, 113), thickness=2)
        handled_image_frame = cv2.putText(img=frame, text=text_draw2, org=draw_text_postion2,
                                          fontFace=font_draw_number, fontScale=1, color=(60, 179, 113), thickness=2)
        # handled_image_frame = cv2.putText(img=frame, text=text_draw3, org=draw_text_postion3,
        #                                   fontFace=font_draw_number, fontScale=1, color=(60, 179, 113), thickness=2)
        cv2.imshow('demo', handled_image_frame)
        cv2.waitKey(1)
    print(fps)
    capture.release()
    cv2.destroyAllWindows()

# 计算叉乘
def intersect(line1, line2, p0, p1):
    cross_product1 = (p0[1] - line1[1]) * (line2[0] - line1[0]) > (line2[1] - line1[1]) * (p0[0] - line1[0])
    cross_product2 = (p1[1] - line1[1]) * (line2[0] - line1[0]) > (line2[1] - line1[1]) * (p1[0] - line1[0])
    return cross_product1 != cross_product2

# 画图
def draw_bboxes(image, bboxes, line_thickness):
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
