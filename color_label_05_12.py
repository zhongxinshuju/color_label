#-*- coding : utf-8 -*-
# coding: utf-8
import json
import pandas as pd
import os
import cv2
import numpy as np
import urllib
from PIL import Image, ImageDraw

color_dict = dict(zip([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150],[(152, 72, 168),(160, 96, 32),(224, 96, 128),(224, 32, 192),(224, 128, 96),(32, 64, 96),(32, 96, 64),(96, 128, 96),(64, 224, 160),(160, 32, 64),(160, 64, 192),(64, 32, 128),(128, 192, 32),(160, 32, 160),(192, 192, 224),(96, 96, 32),(160, 96, 224),(64, 96, 192),(64, 192, 96),(96, 224, 160),(192, 224, 192),(64, 128, 160),(160, 128, 224),(160, 224, 128),(64, 128, 192),(160, 192, 160),(192, 160, 96),(128, 160, 160),(64, 32, 96),(64, 96, 32),(96, 96, 192),(192, 224, 32),(192, 160, 160),(224, 64, 64),(224, 160, 160),(224, 96, 96),(32, 160, 96),(32, 32, 224),(32, 224, 32),(128, 96, 128),(64, 224, 32),(128, 64, 96),(128, 32, 128),(192, 32, 160),(128, 192, 128),(192, 96, 224),(192, 160, 32),(64, 224, 224),(64, 128, 128),(192, 192, 64),(96, 32, 192),(64, 160, 160),(64, 32, 32),(160, 128, 192),(96, 160, 224),(224, 96, 32),(224, 32, 96),(32, 32, 160),(32, 96, 224),(128, 32, 96),(192, 64, 64),(160, 64, 128),(64, 96, 160),(96, 224, 192),(128, 96, 224),(192, 32, 96),(128, 64, 192),(64, 128, 64),(160, 64, 160),(96, 192, 224),(192, 96, 32),(96, 96, 64),(128, 160, 192),(128, 96, 64),(32, 192, 128),(32, 224, 160),(224, 96, 224),(224, 192, 64),(32, 160, 224),(64, 192, 192),(224, 160, 32),(160, 128, 64),(192, 128, 128),(160, 96, 192),(160, 224, 64),(128, 64, 128),(128, 128, 64),(192, 32, 32),(64, 32, 160),(192, 224, 224),(160, 128, 32),(64, 64, 192),(224, 64, 128),(32, 96, 96),(192, 160, 224),(128, 192, 160),(128, 160, 64),(32, 128, 96),(224, 160, 128),(224, 128, 160),(160, 64, 224),(192, 128, 224),(64, 160, 192),(96, 224, 96),(64, 64, 32),(192, 32, 192),(96, 32, 160),(64, 128, 224),(160, 192, 128),(96, 192, 64),(64, 224, 128),(192, 96, 128),(64, 64, 64),(64, 128, 96),(192, 96, 160),(32, 96, 192),(32, 192, 96),(224, 32, 64),(96, 32, 64),(96, 96, 160),(160, 32, 32),(96, 224, 32),(160, 64, 64),(64, 192, 224),(128, 128, 224),(128, 64, 32),(192, 64, 224),(64, 96, 64),(64, 160, 128),(160, 192, 32),(64, 224, 96),(96, 96, 96),(192, 64, 160),(96, 32, 224),(192, 160, 192),(32, 224, 128),(224, 192, 96),(32, 32, 64),(224, 64, 224),(224, 32, 128),(128, 96, 160),(96, 192, 192),(160, 160, 96),(128, 128, 32),(192, 32, 64),(128, 64, 224),(192, 64, 32),(128, 96, 192),(128, 192, 96),(64, 64, 160),(64, 160, 64)]))


dir_csv ="C:\\Users\\Administrator.SKY-20120726UJY\\Desktop\\qinzhen\\csv"
dir_img ="C:\\Users\\Administrator.SKY-20120726UJY\\Desktop\\qinzhen\\real_images"

for cci in os.listdir(dir_csv):
    try:
        data_frame = pd.read_csv( os.path.join(dir_csv, cci),encoding="gb18030")
    except:

        data_frame = pd.read_csv( os.path.join(dir_csv ,cci),encoding="utf-8")

    final_result = data_frame["标注结果"].tolist()
    material_id = data_frame["文件名称"].tolist()
    #打开每一张图片
    for mi, fi in zip(final_result, material_id):
        mi = json.loads(mi)
        fi = str(fi)  
        point_list = []
        x1_list = []
        try:
            im1 = Image.open( os.path.join(dir_img, fi+".png"))
        except:
            im1 = Image.open(os.path.join(dir_img , fi +".jpg"))
        #print(os.path.join(dir_img , fi +".jpg"))
        im2 =ImageDraw.Draw(im1)
        pointl_list = []
       
        img = np.zeros((im1._size[1], im1._size[0], 3), np.uint8)
        cv2.imwrite(fi+".png",img)
        img1 = Image.open( fi + ".png")
        print(fi + ".png")
        #用ImageDraw画出多边形
        img = ImageDraw.Draw(img1)
        mi["zxw"] = []
        test_list = []
        
        for textss in mi['text']:
            y_list = []
            x_list = []
            textl= textss['list']

            #每一个对象由X,Y坐标列表组成
            for listv in textl:
                y_list.append(listv[1])
                x_list.append(listv[0])
            textss['ymax'] = max(y_list)
            textss['ymin'] = min(y_list)
            textss['xmax'] = max(x_list)
            textss['xmin'] = min(x_list)
        #检查是否有重叠区域并保存在test_list表中
        mitext0= mi['text']
        for testv in mitext0:
            for testvl in mitext0:
                if  testvl['xmax'] <= testv['xmax'] and testvl['ymax'] <= testv['ymax'] and testvl['xmin'] >= testv[
                    'xmin'] and testvl['ymin'] >= testv['ymin'] and testv != testvl:
                    test_list.append(testvl)
   
        #对不重叠的区域涂色
        for ll in mitext0:
            listl =ll['list']
            listll = []
            for llll in listl:
                listll.extend(llll)
            labelpath = ll['labelPath'][0]
            color = list(color_dict[int(labelpath['code'])])
            if textl not in test_list:
                im2.polygon(listll, fill=(int(color[0]),int(color[1]),int(color[2])))
                img.polygon(listll, fill=(int(labelpath['code'])))
        #对重叠的区域涂色
        for ll in mitext0:
            listl =ll['list']
            listll = []
            for llll in listl:
                listll.extend(llll)
            labelpath = ll['labelPath'][0]
            color = list(color_dict[int(labelpath['code'])])
            if textl  in test_list:
                im2.polygon(listll, fill=(int(color[0]),int(color[1]),int(color[2])))
                img.polygon(listll, fill=(int(labelpath['code'])))
        
        # 2: rgb label image; 1: black-white  label image
        color = os.path.join(dir_img,"label_color\\")
        try:
          os.makedirs(color)
        except OSError as e:
          pass
        
        heibai = os.path.join(dir_img,"label_images\\")
        try:
          os.makedirs(heibai)
        except OSError as e:
          pass
        
        im1.save(color + fi + ".png")
        img1=img1.convert('L')
        img1.save(heibai + fi + ".png")
        # im3 = cv2.imread(fi + "-2.png")
        # im4=cv2.imread(url2+fi + ".jpg")
        # # im1= cv2.blur(im1, (im1.shape[0], im.shape[1]))
        #
        #
        img2 = Image.open(fi + ".png").convert('L')
        img=np.array(img2)

        print(np.unique(img1))




