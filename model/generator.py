import sys
sys.path.insert(0,'E:\yolov3')

import config 
import utils
import numpy as np 
import cv2
import traceback

def rand(a =0 ,b=1):
    return np.random.rand()*(b-a) + a
def preprocess_gt_boxes(gt_boxes,grid_shapes):

    """
    args:
        gt_boxes : [m][x1,y1,x2,y2,cls]  np.array  m is gt_box numbers ,x1y1 is left up coord,x2y2 is right bottom coord
    return:
        y_true:list of array , like [(52,52,3,(4+4+1)+cls),(26,26,3,(4+4+1)+cls),(13,13,3,(4+4+1)+cls)]
        4+4+1 : 4 is tx ty tw th , second 4 is gt_centerx gt_centery gt_w gt_h  , second 4 is for calc iou 
    """
    y_true = [np.zeros(
                    (grid_shapes[l][0],grid_shapes[l][1],len(config.yolo_layer_anchor[l]),4+4+1+config.num_classes),
                    dtype = 'float32') for l in range(config.num_layers)]
     
    orign_gt_boxes = np.copy(gt_boxes[:,0:4])
    orign_gt_boxes = utils.convert_boxes_to_origin(orign_gt_boxes)

    anchors  = np.array(config.anchors)
    orign_anchors = np.zeros((anchors.shape[0],4))

    #print('anchors:',config.anchors)

    orign_anchors[:,2:4] = anchors[:][:]
    orign_anchors = utils.convert_boxes_to_origin(orign_anchors)

    for id , ogt in enumerate(orign_gt_boxes):
        gt_box = gt_boxes[id]
        #1.gt_box 最大iou的anchor
        iou = utils.cal_iou(ogt,orign_anchors) 
        best_anchor = anchors[np.argmax(iou,axis=-1)]

        #print('gt_box:',gt_box,'ogt_box:',ogt,'best_anchor:',best_anchor,'o_anchors',orign_anchors)

        #2.iou最大的anchor 属于哪一层第几个
        lindx,aindx = config.yolo_anchor_layerIndex[tuple(best_anchor)]
        #3.gtbox在grid_shape中的位置和偏移量tx,ty
        grid_shape = config.stride[lindx]      
        py,px,ty,tx = utils.cal_box_offset_pos(gt_box,grid_shape)
        #4.计算 gt_center_x gt_centery gt_w , gt_h
        gt_w ,gt_h = (gt_box[2]-gt_box[0] , gt_box[3]-gt_box[1])
        gt_center_x = (px + tx ) * grid_shape 
        gt_center_y = (py + ty ) * grid_shape 
        assert gt_w>0 and  gt_h >0  ,r'gt_box w,h <0'
        #4.计算tw,th
        anchor_w , anchor_h  = best_anchor
        tw = np.log(gt_w/anchor_w)
        th = np.log(gt_h/anchor_h)
        cls = int(gt_box[-1])
        aindx = int(aindx)
        lindx = int(lindx)
        y_true[lindx][py,px,aindx,0:4] =(tx,ty,tw,th)
        y_true[lindx][py,px,aindx,4:8] =(gt_center_x,gt_center_y,gt_w,gt_h)
        y_true[lindx][py,px,aindx,8] = 1
        y_true[lindx][py,px,aindx,9+cls] = 1
        
        #print('gt_box;',gt_box,gt_center_x,gt_center_y,gt_w,gt_h)

    return y_true

def get_random_data(annotation_line, input_shape,itter=.2):
    '''random preprocessing for real-time data augmentation
       args: 
           input_shape (h,w)  image shape for train
    '''
    line = annotation_line.split()
    image = cv2.imread(line[0])
    box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    w = image.shape[1]
    h = image.shape[0]
      
    flip = rand()
    if(flip>0.7):
        #左右翻转
        image = image[:,::-1,:]
        box[:,[0,2]] = w - box[:,[2,0]]
    elif (flip>0.4):
        #上下翻转
        image = image[::-1,:,:]
        box[:,[1,3]] = h - box[:,[3,1]]
               
    #resize 
    scaleh = 1.0 * h / input_shape[1]
    scalew = 1.0 * w / input_shape[0]
    image = cv2.resize(image,(input_shape[1],input_shape[0]),interpolation = cv2.INTER_CUBIC)
    box = box.astype(np.float)
    box[:,[0,2]] *= 1.0 /scalew
    box[:,[1,3]] *= 1.0 /scaleh

    #bgr->rgb ->0-1
    return image[:,:,::-1]/255.0, box

class Generator():
    def __init__(self,annotation_path,batch_size,shape,num_classes = config.num_classes, shuffle = True):
        self.annotation_path = annotation_path 
        self.batch_size = batch_size
        self.shuffle =  shuffle
        self.num_classes = num_classes
        self.lines = self.read_lines(self.annotation_path)
        self.batch_idx = utils.BatchIndices(self.num_samples(),self.batch_size,self.shuffle)
        self.shape = shape 

    def num_classes(self):
        return self.num_classes

    def read_lines(self,annatation_path):
        with open(annatation_path) as f:
            lines = f.readlines()
        return lines 

    def num_samples(self):
        return len(self.lines)

    def __next__(self):
        self.idx = next(self.batch_idx)
        try:
            bz = len(self.idx)
            grid_shapes = config.grid_shapes(np.array(self.shape))
            imgs = np.zeros((bz,self.shape[0],self.shape[1],3))
            ys_true =[np.zeros((bz,grid_shapes[l][0],grid_shapes[l][1], \
            len(config.yolo_layer_anchor[l]), 4+4+1+self.num_classes),dtype=np.float) for l in range(config.num_layers)]
            for i,id in enumerate(self.idx):
                img,box = get_random_data(self.lines[id],self.shape)
                y_true = preprocess_gt_boxes(box,grid_shapes)
                imgs[i] = img
                for l in range(config.num_layers):
                    ys_true[l][i]= y_true[l]

            return [imgs,*ys_true],np.zeros((bz))
        except Exception as e :
            pass
            #print(e,self.idx)
            #traceback.print_exc()
            #self.__next__()


#gen = Generator( r'C:\dataset\jinnan2_round1_train_20190305\jinnan2_round1_train_20190305\restricted\train.txt',config.num_classes,(416,416))
#a,b = next(gen)

##true_boxes = np.zeros((2,5))
##true_boxes[0] = [20,20,100,100,0]
##true_boxes[1] = [120,65,220,220,1]


##anchors = np.zeros((9,2))
##anchors[0] = [10,10]
##anchors[1] = [20,20]
##anchors[2] = [30,30]
##anchors[3] = [40,40]
##anchors[4] = [50,50]
##anchors[5] = [60,60]
##anchors[6] = [70,70]
##anchors[7] = [80,80]
##anchors[8] = [90,90]

##num_classes = 2 


##res =  preprocess_gt_boxes(true_boxes)
##res1 = res[0]
##res2 = res[1]
##res3 = res[2]
##res1[res1[:,:,:,4]>0]
##res2[res2[:,:,:,4]>0]
##res3[res3[:,:,:,4]>0]

##np.where(res3[:,:,:,4]>0)

##a  = [0.1875, 0.1875, 0.10536052, 0.10536052]


##b = 