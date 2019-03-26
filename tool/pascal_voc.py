
"""
convert pascal-voc xml to numpy
convert numpy to pascal-voc
"""


import xmltodict
from dicttoxml import dicttoxml
import glob
import os 
import csv 
import numpy as np 
import collections
from xml.dom.minidom import parseString

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

def tree(): return collections.defaultdict(tree)
class Node(object):
    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name
def save_pascal_voc(imgfile,gtboxes):
    xml = tree()
    xml['annotation']['folder'] = os.path.dirname(imgfile)
    xml['annotation']['filename'] = os.path.basename(imgfile).split('.')[0]
    xml['annotation']['path'] = imgfile
    xml['annotation']['source']['dataset'] = 'Unknown'
    xml['annotation']['size']['witdh'] = 3072
    xml['annotation']['size']['height'] = 2048
    xml['annotation']['size']['depth'] = 3
    xml['annotation']['segmented'] = 0
    
    
    for box in gtboxes:
        b = tree()
        b['name'] = box[4]
        b['pose'] = 'Unspcified'
        b['truncate'] = 0
        b['difficult'] = 0
        b['bndbox']['xmin'] = box[0]
        b['bndbox']['ymin'] = box[1]
        b['bndbox']['xmax'] = box[2]
        b['bndbox']['ymax'] = box[3]
        xml['annotation'][Node('object')] = b
        
    xmlpath = os.path.join(xml['annotation']['folder'] ,xml['annotation']['filename']+'.xml')
   
    with open(xmlpath,'w',encoding='utf-8') as f:
        reparsed = parseString(dicttoxml(xml,attr_type=False,root=False))
        f.write(reparsed.toprettyxml(indent=' '*4))

        