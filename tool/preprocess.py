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
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes,dtype = 'float32')
    input_shape = np.array(input_shape,dtype='int32')
    boxes_xy = (true_boxes[...,0:2] + true_boxes[...,2:4]) //2  #中心坐标(x,y)
    boxes_wh = true_boxes[...,2:4] - true_boxes[...,0:2]        #宽高坐标(w,h)
    #比如第0和1位设置为xy，除以416，归一化，第2和3位设置为wh，除以416，归一化，如[0.449, 0.730, 0.016, 0.026, 0.0]
    true_boxes[...,0:2] = boxes_xy / input_shape[::-1]  #input_shape 是hw,这里做个颠倒
    true_boxes[...,2:4] = boxes_wh / input_shape[::-1]

    #m 是batch_size比如为16
    m = true_boxes.shape[0]
    #grid_shape是input_shape等比例降低，即[[13,13], [26,26], [52,52]]；
    grid_shapes = [input_shape // {0:32,1:16,2:8}[l] for l in range(num_layers)]
    #y_true是全0矩阵（np.zeros）列表，即[(16,13,13,3,6), (16,26,26,3,6), (16,52,52,3,6)]
    #这里还不知道len(anchor_mask)要干嘛
    y_true = [np.zeros(
                        (m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask),5+num_classes),
                        dtype = 'float32') for l in range(num_layers)]

    anchors = np.expand_dims(anchors,0)
    anchors_maxes = anchors/2.
    anchors_mins = - anchors_maxes
    valid_mask = boxes_wh[...,0] >0   #将boxes_wh中宽w大于0的位，设为True，即含有box，结构是(m,T)

