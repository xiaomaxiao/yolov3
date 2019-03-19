import sys
sys.path.insert(0,'C:\jianweidata\yolov3')

import config 
import utils
import numpy as np 
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
    orign_anchors[:,2:4] = anchors[:][:]
    orign_anchors = utils.convert_boxes_to_origin(orign_anchors)

    for id , ogt in enumerate(orign_gt_boxes):
        gt_box = gt_boxes[id]
        #1.gt_box 最大iou的anchor
        iou = utils.cal_iou(ogt,orign_anchors) 
        best_anchor = anchors[np.argmax(iou,axis=-1)]
        #2.iou最大的anchor 属于哪一层第几个
        lindx,aindx = config.yolo_anchor_layerIndex[tuple(best_anchor)]
        #3.gtbox在grid_shape中的位置和偏移量tx,ty
        grid_shape = config.stride[lindx]      
        py,px,ty,tx = utils.cal_box_offset_pos(gt_box,grid_shape)
        #4.计算 gt_center_x gt_centery gt_w , gt_h
        gt_w ,gt_h = (gt_box[2]-gt_box[0] , gt_box[3]-gt_box[1])
        gt_center_x = (py + tx ) * stride 
        gt_center_y = (py + ty ) * stride 
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
    return y_true

def get_random_data(annotation_line, input_shape,itter=.2):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    image = cv2.imread(line[0])
    cbox = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    w = image.shape[1]
    h = image.shape[0]
    
    box = np.array([])
    cx1,cy1,cx2,cy2=(0,0,0,0)
       
    #随机缩放
    scale = rand(1-itter,1+itter)
    ih = int(input_shape[0]* scale)
    iw = int(input_shape[1]*scale)
    
    #随机剪切
    for i in range(1000):
        box = np.copy(cbox)
        cx1 = np.random.randint(0,w-iw)
        cy1 = np.random.randint(0,h-ih)
        
        cx2 = cx1 + iw -1
        cy2 = cy1 + ih -1

        #过滤box
        box[:,[0,2]]-=cx1
        box[:,[1,3]]-=cy1
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>=iw] = iw-1
        box[:, 3][box[:, 3]>=ih] = ih-1

        a = box[:,0] >=0
        b = box[:,1] >=0
        c = box[:,2] <iw
        d = box[:,3] <ih    
        ##todo
        ##根据不同的细胞类别决定最小尺寸   
        e = (box[:,2] - box[:,0])>20
        f = (box[:,3] - box[:,1])>20
        box = box[a&b&c&d&e&f]
        if(box.shape[0]>0):
            break

    image_data = image[cy1:cy2+1,cx1:cx2+1,:]
    
    flip = rand()
    if(flip>0.7):
        #左右翻转
        image_data = image_data[:,::-1,:]
        box[:,[0,2]] = iw - box[:,[2,0]]
    elif (flip>0.4):
        #上下翻转
        image_data = image_data[::-1,:,:]
        box[:,[1,3]] = ih - box[:,[3,1]]
               
    #resize 
    image_data = cv2.resize(image_data,(input_shape[1],input_shape[0]),interpolation = cv2.INTER_CUBIC)
    box[:,0:4] *= 1.0 /scale
      
    #bgr->rgb ->0-1
    return image_data[:,:,::-1]/255.0, box

class Generator():
    def __init__(self,annotation_path,batch_size,shape,num_classes = 4, shuffle = True):
        self.annotation_path = annotation_path 
        self.lock = threading.Lock()
        self.batch_size = batch_size
        self.shuffle =  shuffle
        self.num_classes = num_classes
        self.lines = self.read_lines(self.annotation_path)
        self.batch_idx = utils.BatchIndices(self.num_samples,self.batch_size,self.shuffle)
        self.shape = shape 

    def num_classes(self):
        return self.num_classes

    def read_lines(self,annatation_path):
        with open(annotation_path) as f:
            lines = f.readlines()
        return lines 

    def num_samples(self):
        return len(self.lines)

    def __next__(self):
        idx = next(self.batch_idx)
        self.idx = idx
        try:
            bz = len(self.idx)
            imgs = np.zeros((bz,self.shape[0],self.shape[1],3))
            ys_true = [] 
            for id in self.idx:
                img,box = get_random_data(self.annotation_line[id],self.shape)
                y_true = preprocess_gt_boxes(box,config.grid_shapes(self.shape))
                imgs[id] = img
                ys_true.append(y_true)
            return [imgs,*ys_true],np.zeros((batch_size))
        except Exception as e :
            print(e,self.lines[idx])
            self.__next__()





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