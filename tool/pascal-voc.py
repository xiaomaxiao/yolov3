
"""
read pascal-voc xml 
"""


import xmltodict
import glob
import os 
import csv 
import numpy as np 

def read_pascal_voc(path):
    gtboxes=[]
    imgfile = ''
    with open(path,'rb') as f :
       
        xml = xmltodict.parse(f)
        bboxes = xml['annotation']['object']
        if(type(bboxes)!=list):
            x1 = bboxes['bndbox']['xmin']
            y1 = bboxes['bndbox']['ymin']
            x2 = bboxes['bndbox']['xmax']
            y2 = bboxes['bndbox']['ymax']
            name =  bboxes['name']
            gtboxes.append((x1,y1,x2,y2,name))
        else:
            for i in bboxes:
                x1 = i['bndbox']['xmin']
                y1 = i['bndbox']['ymin']
                x2 = i['bndbox']['xmax']
                y2 = i['bndbox']['ymax']
                name = i['name']
                gtboxes.append((x1,y1,x2,y2,name))

        imgfile = xml['annotation']['filename']
    return np.array(gtboxes),imgfile

path = r'C:\jianweidata\yolov3\data'
ann = {}
xmlfiles = glob.glob(os.path.join(path,'*.xml'))
for file in xmlfiles:
    gtboxes , imgfile = read_pascal_voc(file)
    


        