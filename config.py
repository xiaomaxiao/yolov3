import numpy as np 

#每层对应的anchor
yolo_layer_anchor = {
    0:[(19,31),(36,55),(40,29)],
    1:[(52,15),(66,47),(70,103)],
    2:[(86,30),(125,88),(161,51)]
    }

#层数
num_layers = len(yolo_layer_anchor)

anchors = []
for layer in range(num_layers):
    anchors +=yolo_layer_anchor[layer]

num_classes = 5

#anchor对应的哪一层的第几个
"""
例如
{(10, 13): [0, 0],
 (16, 30): [0, 1],
 (30, 61): [1, 0],
 (33, 23): [0, 2],
 (59, 119): [1, 2],
 (62, 45): [1, 1],
 (116, 90): [2, 0],
 (156, 198): [2, 1],
 (373, 326): [2, 2]}
"""
yolo_anchor_layerIndex = {}
for layerindx,_anchors in yolo_layer_anchor.items():
    for id , anchor in enumerate(_anchors):
        yolo_anchor_layerIndex[anchor] = [layerindx,id]

##训练图片大小h,w
#input_shape = np.array([416,416],dtype='int32')

#每层的stride
stride = {
    0:8,
    1:16,
    2:32
    }

#每层的feturemap size  y,x
grid_shapes =lambda input_shape : [input_shape // stride[l] for l in range(num_layers)]


#最大gt_boxes数量
#固定gt_box的长度，算loss中iou的时候方便计算
#max_boxes_num = 100