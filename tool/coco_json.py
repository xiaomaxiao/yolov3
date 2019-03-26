
import json
import os 
from collections import defaultdict

def read_json(file):
    with open(file,'r') as f:
        js = json.load(f)

    imgid_to_imgfile = js['images']
    categories = js['categories']
    annotations = js['annotations']

    data = defaultdict(list)
    #data[imagefile] = gtboxes 
    for ann in annotations:
        img_id = ann['image_id']
        clsid = ann['category_id']
        bbox = ann['bbox']
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2] + x1 
        y2 = bbox[3] + y1 
        imgfile = imgid_to_imgfile[img_id]['file_name']
        data[imgfile].append([x1,y1,x2,y2,clsid])
    return data ,categories


#ann =  read_json(r'C:\dataset\jinnan2_round1_train_20190305\jinnan2_round1_train_20190305\train_no_poly.json')