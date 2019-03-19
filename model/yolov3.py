
import sys
sys.path.insert(0,r'C:\jianweidata\yolov3')

from keras.layers import Conv2D , LeakyReLU , BatchNormalization, ZeroPadding2D , Add ,Input ,Lambda
from keras.layers import UpSampling2D , Concatenate
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf 
from keras.engine.topology import Layer

import config 


class YOLO_V3():
    def __init__(self,input_shape,num_class):
        self.input = Input((None,None,3))
        self.num_class = num_class

    def _conv_2d(self,input,filters,kernel_size=(3,3),strides=(1,1),kernel_regularizer=l2(1e-5),use_bias=True):
        padding = 'valid' if strides==(2,2) else 'same'
        return Conv2D(filters,kernel_size,strides = strides , padding = padding,
                   kernel_regularizer = kernel_regularizer ,use_bias=use_bias)(input)

    def _conv_bn_leaky(self,input,filters,kernel_size=(3,3),strides=(1,1),kernel_regularizer=l2(1e-5)):       
        x = self._conv_2d(input,filters,kernel_size,strides = strides ,
                   kernel_regularizer = kernel_regularizer,use_bias=False)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x 

    def _residual_block(self,input,filters,num_blocks):
        x = ZeroPadding2D(((1,0),(1,0)))(input)
        x = self._conv_bn_leaky(x,filters,(3,3),strides=(2,2))
        for i in range(num_blocks):
            y = self._conv_bn_leaky(x,filters//2,(1,1))
            y = self._conv_bn_leaky(y,filters,(3,3))
            x = Add()([x,y])
        return x 

    def darknet_body(self,input):
        """
        darnet53
        """
        x = self._conv_bn_leaky(input,32,(3,3))
        x = self._residual_block(x,64,1)
        x = self._residual_block(x,128,2)
        x = self._residual_block(x,256,8)
        x = self._residual_block(x,512,8)
        x = self._residual_block(x,1024,4)
        return x 

    def _make_last_layer(self,input,filters,out_fliters):
        
        #convolution Set
        x = self._conv_bn_leaky(input,filters,(1,1))
        x = self._conv_bn_leaky(x,filters*2,(3,3))
        x = self._conv_bn_leaky(x,filters,(1,1))
        x = self._conv_bn_leaky(x,filters*2,(3,3))
        x = self._conv_bn_leaky(x,filters,(1,1))

        #predict
        y = self._conv_bn_leaky(x,filters*2,(3,3))
        y = self._conv_2d(y,out_fliters,(1,1)) 

        return x,y

    def yolo_body(self,input):

        darknet = Model(input,self.darknet_body(input)) 

        num_anchors = [len(config.yolo_layer_anchor[i]) for i in range(config.num_layers)]

        #predict
        x,y3 = self._make_last_layer(darknet.output,512,num_anchors[2]*(self.num_class+5))

        #upsample
        x = self._conv_bn_leaky(x,256,(1,1))
        x = UpSampling2D(2)(x)
        x = Concatenate()([x,darknet.layers[152].output])
        #predict
        x,y2 = self._make_last_layer(x,256,num_anchors[1]*(self.num_class+5))


        #upsample
        x = self._conv_bn_leaky(x,128,(1,1))
        x = UpSampling2D(2)(x)
        x = Concatenate()([x,darknet.layers[92].output])
        #predict
        x,y1 = self._make_last_layer(x,128,num_anchors[0]*(self.num_class+5))

        return Model(input, [y3,y2,y1])

    def yolo_head(self,feats,anchors,stride,num_class,calc_loss=True):
        #reshape to batchsize ,h,w,num_anchors,box_params
        num_anchors = len(anchors)
        anchors_tensor = K.constant(anchors)
        grid_shape = K.shape(feats)[1:3]

        #生成坐标网格xy
        grid_y = K.tile(K.reshape(K.arange(0,grid_shape[0]),[-1,1,1,1]),   # like (13,1,1,1)
                        [1,grid_shape[1],1,1]   # like (13,13,1,1)
                        )
        grid_x = K.tile(K.reshape(K.arange(0,grid_shape[1]),[1,-1,1,1]), # like(1,13,1,1)
                        [grid_shape[0],1,1,1]
                        )
        grid = K.concatenate([grid_x,grid_y],axis=-1)
        
        feats  = K.reshape(feats,[-1,grid_shape[0],grid_shape[1],num_anchors,num_class+5])  # like(-1,13,13,num_anchors,num_class+5)
        grid = K.cast(grid, K.dtype(feats))
        box_center_xy = (K.sigmoid(feats[...,:2])+grid) * stride 
        box_wh = K.exp(feats[...,2:4]) * anchors_tensor 
        box_confidence = K.sigmoid(feats[...,4:5])
        box_class_probs = K.sigmoid(feats[...,5:])

        if(calc_loss):
            return feats,box_center_xy,box_wh
        return box_center_xy,box_wh,box_confidence,box_class_probs
       
    def yolo_losses(self,args):
        yolo_body_output = args[:config.num_layers]
        y_true = args[config.num_layers:]

        loss = 0.0
        for i in range(config.num_layers):
            feats,box_center_xy,box_wh = self.yolo_head(yolo_body_output[i],config.yolo_layer_anchor[config.num_layers-i-1],
                                                          config.stride[i],self.num_class,calc_loss=True)
            input_shape = K.shape(feats)[1:3] * config.stride[i]
            loss += self.yolo_loss(feats,box_center_xy,box_wh,input_shape,y_true[i])
        return loss 

    def yolo_loss(self,feats,box_center_xy,box_wh,input_shape,y_true,ignore_thresh=.5): 
        #object/no object mask 
        y_true = tf.cast(y_true,K.dtype(feats))
        input_shape = tf.cast(input_shape,K.dtype(feats))
        object_mask = y_true[...,8:9]
        no_object_mask = 1 - object_mask

        gt_t_xy = y_true[...,0:2]
        gt_t_wh = y_true[...,2:4]
        gt_center_wh = y_true[...,6:8]
        pred_t_xy = feats[...,0:2]
        pred_t_wh = feats[...,2:4]

        #box_loss_scale 加重小框的权重
        box_loss_scale = 2 - (gt_center_wh[...,0:1]/input_shape[1]) * (y_true[...,1:2]/input_shape[0])
        loss_xy = object_mask * box_loss_scale * K.binary_crossentropy(gt_t_xy,pred_t_wh,from_logits = True)
        loss_wh = 0.5 * (1 - object_mask) * box_loss_scale * K.square(gt_t_wh-pred_t_wh)

        #计算iou 每个cell的最大iou 小于ignore_threshold 则 加入 ignore mask
        #不含物体的网格太多，并不是计算所有的no object网格，而是计算iou低于阈值的
        gt_box = self._trans_box(y_true[...,4:8])
        pred_box = self._trans_box(K.concatenate([box_center_xy,box_wh]))
        iou = self.iou(gt_box,pred_box)
        best_iou = K.max(iou,axis=-1)
        best_iou = K.expand_dims(best_iou,axis=-1)
        ignore_mask = K.cast(best_iou<ignore_thresh,tf.float32)
        loss_confidence = object_mask * K.binary_crossentropy(y_true[...,8:9],feats[...,4:5],from_logits = True) + \
                no_object_mask * ignore_mask * K.binary_crossentropy(y_true[...,8:9],feats[...,4:5],from_logits= True)
        
        loss_cls = object_mask * K.binary_crossentropy(y_true[...,9:],feats[...,5:],from_logits=True)
        
        total_loss = K.concatenate([loss_xy,loss_wh, loss_confidence, loss_cls], axis=-1)
        total_loss = K.mean(K.mean(total_loss, axis=[1, 2, 3, 4]))

        return total_loss

    def _trans_box(self,box):
        b_xy = box[..., 0:2]
        b_wh = box[..., 2:4]
        b_wh_half = b_wh/2.
        b_mins = b_xy - b_wh_half
        b_maxes = b_xy + b_wh_half
        return K.concatenate([b_mins,b_maxes],axis=-1)

    def iou(self,box1,box2):
        x1 = K.maximum(box1[...,0],box2[...,0])
        x2 = K.minimum(box1[...,2],box2[...,2])
        y1 = K.maximum(box1[...,1],box2[...,1])
        y2 = K.minimum(box1[...,3],box2[...,3])

        box1_area = (box1[...,0]-box1[...,2]) * (box1[...,1] - box1[...,3])
        box2_area = (box2[...,0] - box2[...,2]) * (box2[...,1] - box2[...,3])

        intersection = K.maximum(x2-x1,0) * K.maximum(y2-y1,0)
        iou = intersection / (box1_area + box2_area[:] - intersection[:] + 1e-6)
        return iou

    def create_yolo(self):
        input_shape = K.shape(self.input)[1:3]
        #grid_shapes = config.grid_shapes(input_shape)
        y_true = [Input(shape=(None,None, \
            len(config.yolo_layer_anchor[l]), 4+4+1+self.num_class)) for l in range(config.num_layers)]
        yolo_body = self.yolo_body(self.input)      
        model_loss = Lambda(self.yolo_losses, output_shape=(1,), name='yolo_loss')(
            [*yolo_body.output,*y_true])     
        model = Model([self.input,*y_true],model_loss)
        return yolo_body,model 

    def eval_yolo(self,yolo_body_output,score_thresh=0.3,iou_threshold = 0.45 , max_pre_cls = 100):      
        #to do tf.map_fn parallel
        boxes = []
        scores = []      
        for i in range(config.num_layers):
            box_center_xy,box_wh,box_confidence,box_class_probs = self.yolo_head(yolo_body_output[config.num_layers-i-1],config.yolo_layer_anchor[i],
                                                                                        config.stride[i],self.num_class,calc_loss=False)
            #box_center_xy = tf.Print(box_center_xy,['box_center_xy',box_center_xy])
            box = self._trans_box(K.concatenate([box_center_xy,box_wh],axis=-1))
            score = box_confidence * box_class_probs
            boxes.append(K.reshape(box,(-1,4)))
            scores.append(K.reshape(score,(-1,self.num_class)))

        boxes = K.concatenate(boxes,axis=0)
        scores = K.concatenate(scores,axis=0)

        #保留得分大于阈值
        boxes_=[]
        scores_=[]
        classes_=[]
        mask = scores > score_thresh
        for c in range(self.num_class):
            c_boxes = tf.boolean_mask(boxes,mask[:,c])
            c_scores = tf.boolean_mask(scores[:,c],mask[:,c])

            nms_index = tf.image.non_max_suppression(
                                    c_boxes, c_scores, max_pre_cls, iou_threshold)

            c_boxes = K.gather(c_boxes, nms_index)
            c_scores = K.gather(c_scores, nms_index)
            classes = K.ones_like(c_scores, 'int32') * c

            boxes_.append(c_boxes)
            scores_.append(c_scores)
            classes_.append(classes)

        boxes_ = K.concatenate(boxes_, axis=0)
        scores_ = K.concatenate(scores_, axis=0)
        classes_ = K.concatenate(classes_, axis=0)

        return boxes_, scores_, classes_

    def __test_gridxy(self):
        import numpy as np 
        grid_shape = (13,13)
        grid_y = np.tile(np.reshape(np.arange(0,grid_shape[0]),[-1,1,1,1]),   # like (13,1,1,1)
                [1,grid_shape[1],1,1]   # like (13,13,1,1)
                )
        grid_x = np.tile(np.reshape(np.arange(0,grid_shape[1]),[1,-1,1,1]), # like(1,13,1,1)
                        [grid_shape[0],1,1,1]  # like (13,13,1,1)
                        )
        grid_xy = np.concatenate((grid_x,grid_y),axis = -1)

#yolo = YOLO_V3((416,416,3),config.num_classes)
#yolo_body,model = yolo.create_yolo()



#def  test_eval():
#    ## test yolov3 
#    sess = K.get_session()
#    yolo = YOLO_V3((416,416,3),80)
#    yolo_body = yolo.yolo_body(yolo.input)
#    #yolo_body.summary()
#    yolo_body.load_weights(r'C:\jianweidata\yolov3\model_data\yolo_416.h5')
#    boxes_, scores_, classes_ = yolo.eval_yolo(yolo_body.output)
#    import cv2
#    import numpy as np 
#    img = cv2.imread(r'C:\jianweidata\yolov3\kite.jpg')
#    img = cv2.resize(img,(544,544))
#    img2 = np.expand_dims(img,0)
#    img2 = img2/255.0

#    out_boxes, out_scores, out_classes = sess.run(
#        [boxes_, scores_, classes_],
#        feed_dict={
#            yolo.input: img2,
#            K.learning_phase(): 0
#        })

#    for idx,box in enumerate(out_boxes):
#        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(255,0,0),2)  

#    cv2.imshow('image', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


