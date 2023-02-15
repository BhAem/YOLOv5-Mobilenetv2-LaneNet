from numba import jit
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


# 卡尔曼滤波
class KalmanBoxTracker(object):
    num_of_kbt = 0  # 记录卡尔曼滤波器的创建个数
    def __init__(self, box):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)  
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])  
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])  
        self.kf.R[2:, 2:] *= 10.  
        self.kf.P[4:, 4:] *= 1000. 
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01 
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_xyxy_to_xysr(box) 
        self.time_after_update = 0  
        self.id = KalmanBoxTracker.num_of_kbt
        self.consecutive_history = []
        self.consecutive_hits = 0
        self.age = 0
        self.cls_id = box[-1]
        KalmanBoxTracker.num_of_kbt += 1

    # 使用yolov5s-mobilenetv2检测到的目标框box更新状态变量
    def update(self, box):
        self.time_after_update = 0
        self.consecutive_history = []
        self.consecutive_hits += 1
        self.kf.update(convert_xyxy_to_xysr(box))

    # 使用当前对象的卡尔曼滤波器链进行目标框的预测
    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_after_update > 0:
            self.consecutive_hits = 0
        self.time_after_update += 1
        self.consecutive_history.append(convert_xysr_to_xyxy(self.kf.x))
        return self.consecutive_history[-1]

    def get_state(self):
        return convert_xysr_to_xyxy(self.kf.x)


class Sort(object):

    def __init__(self, max_age=1, min_hits=3):
        self.max_age = max_age  # 目标未被检测到的帧数，超过之后会被删
        self.min_hits = min_hits  # 目标连续命中的最小次数，小于该次数update函数不返回该目标的KalmanBoxTracker卡尔曼滤波对象
        self.trackers = []  # 卡尔曼滤波跟踪器链，存储多个卡尔曼滤波对象
        self.frame_count = 0  # 帧计数

    def update(self, dets):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))  # 存储跟踪器的预测
        to_del = []  # 存储要删除的目标框
        ret = []  # 存储要返回的追踪目标框
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 将目标检测框与卡尔曼滤波器预测的跟踪框关联获取跟踪成功的目标，新增的目标，离开画面的目标
        matched, unmatched_dets, unmatched_trks = association(dets, trks)

        # 将跟踪成功的目标框更新到对应的卡尔曼滤波器
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                # 使用观测的边界框更新状态向量
                trk.update(dets[d, :][0])

        # 为新增的目标创建新的卡尔曼滤波器对象进行跟踪
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_after_update < 1) and (trk.consecutive_hits >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1], [trk.cls_id])).reshape(1, -1))
            i -= 1
            if trk.time_after_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


# 进行iou计算
@jit
def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
              - wh)
    return o

# 使用匈牙利算法将卡尔曼滤波的预测框和目标检测的检测框
# 进行IOU匹配来计算相似度从而进行关联匹配
def association(detections, trackers, iou_threshold=0.3):
    if (len(trackers) == 0) or (len(detections) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    result = linear_sum_assignment(-iou_matrix)
    matched_indices = np.array(list(zip(*result)))

    # 记录未匹配的检测框
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    # 记录未匹配的跟踪框
    unmatched_tracks = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_tracks.append(t)

    # 将匹配成功的跟踪框放入matches中
    matched_tracks = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matched_tracks.append(m.reshape(1, 2))

    if len(matched_tracks) == 0:
        matched_tracks = np.empty((0, 2), dtype=int)
    else:
        matched_tracks = np.concatenate(matched_tracks, axis=0)
    return matched_tracks, np.array(unmatched_detections), np.array(unmatched_tracks)


# [x1,y1,x2,y2] -> [x,y,s,r]，其中x、y是框的中心坐标点，s是面积，r是宽高比w/h
def convert_xyxy_to_xysr(box):
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = box[0] + w / 2.
    y = box[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

# [x,y,s,r] -> [x1,y1,x2,y2]
def convert_xysr_to_xyxy(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2.,
                         x[1] - h / 2.,
                         x[0] + w / 2.,
                         x[1] + h / 2.]
                        ).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.,
                         x[1] - h / 2.,
                         x[0] + w / 2.,
                         x[1] + h / 2.,
                         score]).reshape((1, 5))