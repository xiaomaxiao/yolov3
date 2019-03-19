import numpy as np 


def preprocess_true_boxes(true_boxes,input_shape,anchors,num_classes):
    '''
     preprocess true boxes to traning input format

     Parameters:
        true_boxes: array, shape=(m, T, 5) , m bathcsize , T 框的个数
            Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
        input_shape: array-like, hw, multiples of 32
        anchors: array, shape=(N, 2), wh
        num_classes: integer

    Returns
        y_true: list of array, shape like yolo_outputs, xywh are reletive value
    '''
    assert(true_boxes[...,4] < num_classes).all(),'class id must be less than num_classes'
    num_layers = len(anchors)//3
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]

    true_boxes = np.array(true_boxes,dtype = 'float32')
    input_shape = np.array(input_shape,dtype='int32')
    boxes_xy = (true_boxes[...,0:2] + true_boxes[...,2:4]) //2  #中心坐标(x,y)
    boxes_wh = true_boxes[...,2:4] - true_boxes[...,0:2]        #宽高(w,h)
    #比如第0和1位设置为xy，除以416，归一化，第2和3位设置为wh，除以416，归一化，如[0.449, 0.730, 0.016, 0.026, 0.0]
    true_boxes[...,0:2] = boxes_xy / input_shape[::-1]  #input_shape 是hw,这里做个颠倒
    true_boxes[...,2:4] = boxes_wh / input_shape[::-1]

    #m 是batch_size比如为16
    m = true_boxes.shape[0]
    #grid_shape是input_shape等比例降低，如412的图 是[[13,13], [26,26], [52,52]]；
    grid_shapes = [input_shape // {0:32,1:16,2:8}[l] for l in range(num_layers)]
    #y_true是全0矩阵（np.zeros）列表，即[(16,13,13,3,6), (16,26,26,3,6), (16,52,52,3,6)]
    #这里还不知道len(anchor_mask)要干嘛
    y_true = [np.zeros(
                        (m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask),5+num_classes),
                        dtype = 'float32') for l in range(num_layers)]

    anchors = np.expand_dims(anchors,0)  #由(9,2)转为(1,9,2)；
    #算anchors_maxes mins 是为了计算IOU用的
    #以0点为原点(-2/x,2/x)这样搞的最大最小值
    anchors_maxes = anchors/2.
    anchors_mins = - anchors_maxes

    #将boxes_wh中宽w大于0的位，设为True，即含有box，结构是(m,T)
    #作者这样设定是因为他定义了一个true_boxes的最大量比如20个，不足20个的用0填充了，超过20的截取了，为什么这么搞呢？
    #可能是他觉得一张图的标记的时候目标不会多于20个？
    valid_mask = boxes_wh[...,0] >0   

    for b in range(m):
        wh = boxes_wh[b,valid_mask]
        if len(wh)==0: continue 

        wh = np.expand_dims(wh,-2)  #是wh倒数第2个添加1位，如(*,2)->(*,1,2)  *是框的个数
        box_maxes = wh/2.
        box_mins = - box_maxes

        #计算IOU,每个标注框box与9个anchor box 的Iou值(*,9)
        intersect_mins  = np.maximum(box_mins,anchors_mins)
        intersect_maxes = np.minimum(box_maxes,anchors_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins,0.)
        intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
        box_area = anchors[...,0] * anchors[...,1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        #为每一个true box 找最大IOU的anchor
        best_anchor = np.argmax(iou,axis = -1)
        for t,n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[b,t,0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,0] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t,4].astype('int32')
                    y_true[l][b,j,i,k,0:4] = true_boxes[b,t,0:4]
                    y_true[l][b,j,i,k,4] = 1 
                    y_true[l][b,j,i,k,5+c] = 1
        return y_true
