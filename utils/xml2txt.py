import os
import shutil
import xml.dom.minidom as xml

#train
pic_root = "D:\\winterholiday\\UA-DETRAC\\Insight-MVT_Annotation_Train"
new_pic_root = "D:\\winterholiday\\UA-DETRAC2\\images\\train"

xml_root = "D:\\winterholiday\\UA-DETRAC\\DETRAC-Train-Annotations-XML"
new_label_root = "D:\\winterholiday\\UA-DETRAC2\\labels\\train"

# #test
# pic_root = "D:\\winterholiday\\UA-DETRAC\\Insight-MVT_Annotation_Test"
# new_pic_root = "D:\\winterholiday\\UA-DETRAC2\\images\\test"
#
# xml_root = "D:\\winterholiday\\UA-DETRAC\\DETRAC-Test-Annotations-XML"
# new_label_root = "D:\\winterholiday\\UA-DETRAC2\\labels\\test"

seq_width = 960
seq_height = 540

MVI_xxx = [s for s in os.listdir(pic_root)]


for mvi in MVI_xxx:

    name_of_txt = [s.split('.')[0] for s in os.listdir(os.path.join(pic_root, mvi))]

    # move pic from old path to new path
    filelist = os.listdir(os.path.join(pic_root, mvi))
    for file in filelist:
        src = os.path.join(os.path.join(pic_root, mvi), file)
        dst = os.path.join(new_pic_root, file)
        shutil.copy(src, dst)

    xmlfile = os.path.join(xml_root, mvi + '.xml')
    filehandle = open(xmlfile,'rb')
    filecontent = filehandle.read()
    if (filehandle != None):
        filehandle.close()
    if (filecontent != None):
        filecontent.decode("utf-8", "ignore")

    dom = xml.parseString(filecontent)
    root = dom.getElementsByTagName('sequence')[0]
    if root.hasAttribute("name"):
        seq_name = root.getAttribute("name")        #!!
    # 获取所有的frame
    frames = root.getElementsByTagName('frame')

    cnt = 0
    for frame in frames:

        vehicle_info = []

        if frame.hasAttribute("num"):
            frame_num = int(frame.getAttribute("num"))  #!!

        # 获取一帧里面所有的target
        target_list = frame.getElementsByTagName('target_list')[0]
        targets = target_list.getElementsByTagName('target')

        for target in targets:
            if target.hasAttribute("id"):
                tar_id = int(target.getAttribute("id"))
                box = target.getElementsByTagName('box')[0]
                left = 0
                top = 0
                width = 0
                height = 0
                if box.hasAttribute("left"):
                    left = box.getAttribute("left")
                if box.hasAttribute("top"):
                    top = box.getAttribute("top")
                if box.hasAttribute("width"):
                    width = box.getAttribute("width")
                if box.hasAttribute("height"):
                    height = box.getAttribute("height")
                x = float(left) + float(width) / 2
                y = float(top) + float(height) / 2
                attribute = target.getElementsByTagName('attribute')[0]
                label = 0
                # if attribute.hasAttribute("vehicle_type"):
                #     type = attribute.getAttribute("vehicle_type")
                    # if type == "others":
                    #     label = 2
                    # if type == "car":
                    #     label = 0
                    # if type == "van":
                    #     label = 0
                    # if type == "bus":
                    #     label = 1
                vehicle_info.append([0, x, y, float(width), float(height)])

        for label, x, y, w, h in vehicle_info:
            label = int(label)
            label_fpath = os.path.join(new_label_root, name_of_txt[cnt] + '.txt')
            label_str = '{:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(label, float(x)/seq_width, float(y)/seq_height, float(w)/seq_width, float(h)/seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)
        cnt += 1






















