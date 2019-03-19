
import config 
import numpy as np

class BatchIndices():
    def __init__(self,total,batchsize,trainable=True):
        self.n = total
        self.bs = batchsize
        self.shuffle = trainable
        self.lock = threading.Lock()
        self.reset()
    def reset(self):
        self.index = np.random.permutation(self.n) if self.shuffle==True else np.arange(0,self.n)
        self.curr = 0
    
    def __next__(self):
        with self.lock:
            if self.curr >=self.n:
                self.reset()
            rn = min(self.bs,self.n - self.curr)
            res = self.index[self.curr:self.curr+rn]
            self.curr += rn
            return res

def cal_box_offset_pos(gt_box,stride):
    """calculate the postion and offset of gt_box in the grid
    args:
        gt_box:[x1,y1,x2,y2]  np.array
        gird_shape [grid_y,grid_x]  np.array

    return :
        pos (py,px,offsety,offsetx)
    """
    x1,y1,x2,y2,_ = gt_box
    centerx = (x2 + x1) /2. /stride
    centery = (y2 + y1) /2. /stride
    py = np.floor(centery)
    px = np.floor(centerx)
    return int(py),int(px),centery-py ,centerx -px 



def cal_iou(box1,boxes2):
    """ calculate the intersection of boxe1 and boxes2
    
    args:
        box1:   [x1,y1,x2,y2]  np.array
        boxes2: [M][x1,y1,x2,y2]  np.array
    return:
        iou: [M]       np.array
    """
    x1 = np.maximum(box1[0],boxes2[:,0])
    x2 = np.minimum(box1[2],boxes2[:,2])
    y1 = np.maximum(box1[1],boxes2[:,1])
    y2 = np.minimum(box1[3],boxes2[:,3])

    box1_area = (box1[0]-box1[2]) * (box1[1] - box1[3])
    boxes2_area = (boxes2[:,0] - boxes2[:,2]) * (boxes2[:,1] - boxes2[:,3])

    intersection = np.maximum(x2-x1,0) * np.maximum(y2-y1,0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
    return iou


def convert_boxes_to_origin(boxes):
    """
    Convert the center point of the box to (0,0)
    """
    center_xy = (boxes[...,0:2] + boxes[...,2:4]) //2  #中心坐标(x,y)
    boxes[...,0:2] -= center_xy
    boxes[...,2:4] -= center_xy

    return boxes