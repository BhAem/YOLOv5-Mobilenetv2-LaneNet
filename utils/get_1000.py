import os
import shutil
import xml.dom.minidom as xml

# train
# pic_root = "D:\\winter\\UA_DETRAC_yolo_data\\images\\train"
# new_pic_root = "D:\\winter\\UA_DETRAC_yolo_data2\\images\\"
#
# txt_root = "D:\\winter\\UA_DETRAC_yolo_data\\labels\\train"
# new_label_root = "D:\\winter\\UA_DETRAC_yolo_data2\\labels\\"

num = 0.8
num1 = 0.75
limit = 181
#test
pic_root = "D:\\winter\\YOLOv5-Mobilenetv2\\test_ua_detrac_img\\"+str(num)+"\\images"
pic_root2 = "D:\\winter\\YOLOv5-Mobilenetv2\\test_ua_detrac_img\\total\\images"
pic_root3 = "D:\\winter\\YOLOv5-Mobilenetv2\\test_ua_detrac_img\\"+str(num1)+"\\images"

txt_root = "D:\\winter\\YOLOv5-Mobilenetv2\\test_ua_detrac_img\\"+str(num)+"\\labels"
txt_root2 = "D:\\winter\\YOLOv5-Mobilenetv2\\test_ua_detrac_img\\total\\labels"
txt_root3 = "D:\\winter\\YOLOv5-Mobilenetv2\\test_ua_detrac_img\\"+str(num1)+"\\labels"
# pic_root = "D:\\winter\\YOLOv5-Mobilenetv2\\UA_DETRAC_yolo_data\\images\\test"
# pic_root2 = "D:\\winter\\YOLOv5-Mobilenetv2\\test_ua_detrac_img\\total\\images"
#
# txt_root = "D:\\winter\\YOLOv5-Mobilenetv2\\UA_DETRAC_yolo_data\\labels\\test"
# txt_root2 = "D:\\winter\\YOLOv5-Mobilenetv2\\test_ua_detrac_img\\total\\labels"

imgs = [s.split('.')[0] for s in os.listdir(pic_root)]
imgs2 = [s.split('.')[0] for s in os.listdir(pic_root3)]
print(len(imgs2))
txts = [s.split('.')[0] for s in os.listdir(txt_root)]
txts2 = [s.split('.')[0] for s in os.listdir(txt_root3)]
#
# cnt = 0
# for img in imgs:
#     if img in imgs2:
#         src2 = os.path.join(txt_root, img + '.txt')
#         dst2 = os.path.join(txt_root2, img + '.txt')
#         shutil.copy(dst2, src2)
#         cnt += 1
#         print(cnt)


cnt = 0
for img in imgs:

    if img not in txts:
        continue
    if img in imgs2:
        continue
    src1 = os.path.join(pic_root, img + '.jpg')
    dst1 = os.path.join(pic_root2, img + '.jpg')
    shutil.copy(src1, dst1)

    src2 = os.path.join(txt_root, img + '.txt')
    dst2 = os.path.join(txt_root2, img + '.txt')
    shutil.copy(src2, dst2)
    cnt += 1
    print(cnt)
    if cnt == limit:
        break
# print(cnt)





# for img in imgs2:
#     # if cnt % 400 != 0:
#     #     continue
#     # if img not in txts:
#     #     continue
#     # print(img)
#     if img not in txts2:
#         # src1 = os.path.join(pic_root, img + '.jpg')
#         # dst1 = os.path.join(pic_root2, img + '.jpg')
#         # shutil.copy(src1, dst1)
#         #
#         # src2 = os.path.join(txt_root, img + '.txt')
#         # dst2 = os.path.join(txt_root2, img + '.txt')
#         # shutil.copy(src2, dst2)
#         # cnt += 1
#         print(img)

    # if cnt == 200:
    #     break
    # src = os.path.join(pic_root, img+'.jpg')
    # dst = os.path.join(new_pic_root, img+'.jpg')
    # src1 = os.path.join(txt_root, img+'.txt')
    # dst1 = os.path.join(new_label_root, img+'.txt')
    # shutil.copy(src, dst)
    # shutil.copy(src1, dst1)