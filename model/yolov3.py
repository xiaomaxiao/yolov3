
from keras.layers import Conv2D , LeakyReLU , BatchNormalization, ZeroPadding2D , Add
from keras.layers import UpSampling2D , Concatenate
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf 

class YOLO_V3():
    def __init__(self, *args, **kwargs):
        pass


    def _conv_bn_leaky(self,input,filters,kernel_size=(3,3),strides=(1,1),kernel_regularizer=l2(1e-5)):
        
        padding = 'valid' if strides==(2,2) else 'same'
        
        x = Conv2D(filters,kernel_size,strides = strides , padding = padding,
                   kernel_regularizer = kernel_regularizer)(input)
        x = BatchNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x 

    def _residual_block(self,input,fliters,num_blocks):
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
        x = self._conv_bn_leaky(32,(3,3))(input)
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
        y = self.conv2d(y,out_fliters,(1,1))  #待写
        return x,y

    def yolo_body(self,input,num_anchors,num_class):
        darknet = Model(input,self.darknet_body(input)) 
        #predict
        x,y1 = self._make_last_layer(darknet.output,512,num_anchors*(num_class+5))

        #upsample
        x = self._conv_bn_leaky(x,256,(1,1))
        x = UpSampling2D(2)(x)
        x = Concatenate()([x,darknet.layers[152].output])
        #predict
        x,y2 = self._make_last_layer(x,256,num_anchors*(num_class+5))


        #upsample
        x = self._conv_bn_leaky(x,128,(1,1))
        x = UpSampling2D(2)(x)
        x = Concatenate()([x,darknet.layers[92].output])
        #predict
        x,y3 = self._make_last_layer(x,128,num_anchors*(num_class+5))

        return Model(input,[y1,y2,y3])


    def yolo_head(self,feats,anchors,num_classes,input_shape,calc_loss = False):
        '''
        https://mp.weixin.qq.com/s/4L9E4WGSh0hzlD303036bQ
        '''
        num_anchors = len(anchors)
        #reshape to batch,height,width,num_anchors,box_params
        anchors_tensor= K.reshape(K.constant(anchors),[1,1,1,num_anchors,2]) #(1,1,1,3,2)

        grid_shape = K.shape(feats)[1:3] #height,width即预测图feats的第1~2位，如13x13；
        grid_y = K.tile(K.reshape(K.arange(0,stop=grid_shape[0]),[-1,1,1,1]),#(13,1,1,1)
                        [1,grid_shape[1],1,1]) #(13,13,1,1)

        grid_x = K.tile(K.reshape(K.arange(0,stop=grid_shape[1]),[1,-1,1,1]),  #(1,13,1,1)
                                  [grid_shape[0],1,1,1])  #(13,13,1,1)
        grid = K.concatenate([grid_x, grid_y])       #(13,13,1,2)
        gird = K.cast(grid,K.dtype(feats))

        feats = K.reshape(feats,[-1,grid_shape[0],gird_shape[1],num_anchors,num_classes+5])

        
        box_xy = (K.sigmoid(feats[...,:2]) + grid) / K.cast(grid_shape[::-1],K.dtype(feats))#(1,13,13,3,2)
        box_wh = K.exp(feats[...,2:4]) * anchors_tensor / K.cast(input_shape[::-1],K.dtype(feats))#(1,13,13,3,2)
        box_confidence = K.sigmoid(feats[...,4:5])
        box_class_prods = K.sigmoid(feats[...,5:])

        if calc_loss ==True:
            return grid,feats,box_xy,box_wh
        return box_xy,box_wh,box_confidence,box_class_prods
    
    def box_iou(b1,b2):
        '''Return iou tensor

        Parameters
        ----------
        b1: tensor, shape=(i1,...,iN, 4), xywh
        b2: tensor, shape=(j, 4), xywh

        Returns
        -------
        iou: tensor, shape=(i1,...,iN, j)

        '''
        b1 = K.expand_dims(b1,-2)
        b1_xy = b1[...,:2]
        b1_wh = b1[...,2:4]
        b1_wh_half = b1_wh /2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half

        b2 = K.expand_dims(b1,0)
        b2_xy = b2[...,:2]
        b2_wh = b2[...,2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half
        
        intersect_mins = K.maximum(b1_mins,b2_mins)
        intersect_maxes = K.minimum(b1_maxes,b2_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        iou = intersect_area / (b1_area + b2_area - intersect_area)

        return iou

    def yolo_loss(self,yolo_outputs,y_true,anchors,num_classes,ignore_thresh = 0.5 ,print_loss = False):
        '''Return yolo_loss tensor

        Parameters
        ----------
        yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
        y_true: list of array, the output of preprocess_true_boxes
        anchors: array, shape=(N, 2), wh
        num_classes: integer
        ignore_thresh: float, the iou threshold whether to ignore object confidence loss

        Returns
        -------
        loss: tensor, shape=(1,)

        '''
        num_layers = len(anchors)//3
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [0,1,2]]
        # input_shape是输出的尺寸*32, 就是原始的输入尺寸，[1:3]是尺寸的位置，即416x416
        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32,K.dtype(y_true[0]))
        # 每个网络的尺寸，组成列表
        grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3],K.dtype(y_true[0])) for l in range(num_layers)]

        m = K.shape(yolo_outputs[0])[0] # batch size 
        mf = K.cast(m,K.dtype(yolo_outputs[0]))
        loss = 0.0

        for l in range(num_layers):
            object_mask = y_true[l][...,4:5]
            true_class_probs = y_true[l][...,5:]

            grid,raw_pred,pred_xy,pred_wh = self.yolo_head(yolo_outputs[i],
                    anchors[anchor_mask[l]],num_classes,input_shape,calc_loss = True)
            pred_box = K.concatenate([pred_xy,pred_wh])

            #darknet raw box to calculate loss.
            raw_true_xy = y_true[l][...,:2] * grid_shapes[l][::-1] - grid
            raw_true_wh = K.log(y_true[l][...,2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
            raw_true_wh = K.switch(object_mask,raw_true_wh,K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
            box_loss_scale = 2 - y_true[l][...,2:3] * y_true[l][...,3:4]

            #find ingore mask,iterate over each of batch
            ignore_mask = tf.TensorArray(K.dtype(y_true[0]),size =1,dynamic_size = True)
            object_mask_bool = K.cast(object_mask,'bool')

            def loop_body(b,ignore_mask):
                true_box = tf.boolean_mask(y_true[l][b,...,0:4],object_mask_bool[b,...,0])
                iou = box_iou(pred_box[b],true_box)
                best_iou = K.max(iou,axis=-1)
                ignore_mask = ingore_mask.write(b,K.cast(best_iou < ignore_thresh),K.dtype(true_box))
                return b+1,ignore_mask
            _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = K.expand_dims(ignore_mask, -1)

            #binary_crossentropy 是 sigmoid_cross_entropy_with_logits
            #这里raw_pred是feats 还没有进行sigmoid，然后用的交叉熵，而没有用 平方差，
            #为什么没有用平方差呢？需要查查 交叉熵和方差代价函数的不同
            xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
            wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])