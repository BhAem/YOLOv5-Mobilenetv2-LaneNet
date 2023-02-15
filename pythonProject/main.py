# 引用OpenCv2
import cv2

# 定义打开视频对象
cap = cv2.VideoCapture("input4.mp4")
# 读取一帧
ret, frame = cap.read()
# 视频写入格式
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# 视频输出对象
out = cv2.VideoWriter('camera_test.avi', fourcc, fps, size)
# 循环读取、写入
while ret:
    # 读取一帧
    ret, frame = cap.read()
    if frame is None:
        print('read frame is err!')
        continue
    # 显示一帧
    cv2.imshow("frame", frame)

    # 按键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 保存一帧
    out.write(frame)

# 释放窗口32222222222222222222222
cv2.destroyAllWindows()
# 视频写入结束
cap.release()
