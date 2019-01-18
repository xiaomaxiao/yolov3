
from keras.layers import Conv2D , LeakyReLU , BatchNormalization, ZeroPadding2D , Add
from keras.layers import UpSampling2D , Concatenate
from keras.models import Model
from keras.regularizers import l2

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
        y = self._conv_bn_leaky(y,out_fliters,(1,1))
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





        