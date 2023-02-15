import os
import shutil
import xml.dom.minidom as xml

# train
pic_root = "D:\\winter\\YOLOv5-Mobilenetv2\\UA_DETRAC_yolo_data\\images\\test"
new_pic_root = "D:\\winter\\YOLOv5-Mobilenetv2\\UA_DETRAC_yolo_data2\\images\\"

txt_root = "D:\\winter\\YOLOv5-Mobilenetv2\\UA_DETRAC_yolo_data\\labels\\test"
new_label_root = "D:\\winter\\YOLOv5-Mobilenetv2\\UA_DETRAC_yolo_data2\\labels\\"

# #test
# pic_root = "D:\\winter\\UA_DETRAC_yolo_data\\images\\test"
# new_pic_root = "D:\\winter\\UA_DETRAC_yolo_data2\\images\\"
#
# txt_root = "D:\\winter\\UA_DETRAC_yolo_data\\labels\\test"
# new_label_root = "D:\\winter\\UA_DETRAC_yolo_data2\\labels\\"

imgs = [s.split('.')[0] for s in os.listdir(pic_root)]
print(len(imgs))
# txts = [s.split('.')[0] for s in os.listdir(txt_root)]
# cnt = 0
# for img in imgs:
#     cnt += 1
#     # if cnt % 10 != 0:
#     #     continue
#     if img not in txts:
#         continue
#     print(img)
#     src = os.path.join(pic_root, img+'.jpg')
#     dst = os.path.join(new_pic_root, img+'.jpg')
#     src1 = os.path.join(txt_root, img+'.txt')
#     dst1 = os.path.join(new_label_root, img+'.txt')
#     shutil.copy(src, dst)
#     shutil.copy(src1, dst1)


# imgs = [s.split('.')[0] for s in os.listdir(pic_root)]
# print(len(imgs))
# txts = [s.split('.')[0] for s in os.listdir(txt_root)]
# cnt = 0
# for img in imgs:
#     cnt += 1
#     # if cnt % 10 != 0:
#     #     continue
#     if img not in txts:
#         continue
#     print(img)
#     src = os.path.join(pic_root, img+'.jpg')
#     dst = os.path.join(new_pic_root, img+'.jpg')
#     src1 = os.path.join(txt_root, img+'.txt')
#     dst1 = os.path.join(new_label_root, img+'.txt')
#     shutil.copy(src, dst)
#     shutil.copy(src1, dst1)