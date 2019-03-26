

# In[2]:

import tensorflow as tf
import keras.backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement=True
session = tf.Session(config=config)
K.set_session(session)


# In[3]:

# from tensorflow.python import debug as tf_debug
# K.set_session(
#     tf_debug.TensorBoardDebugWrapperSession(tf.Session(), "DESKTOP-NAQHFD5:6006"))


# In[4]:

from model.yolov3 import YOLO_V3
import config 


# In[5]:

input_shape = (416,416,3)
with tf.device('/cpu:0'):
    yolo = YOLO_V3(input_shape,config.num_classes)
    yolo_body,model = yolo.create_yolo()

yolo_body.load_weights(r'E:\yolov3\model_data\yolo_416.h5', by_name=True,skip_mismatch=True)
for i in range(len(yolo_body.layers)-3):  
    yolo_body.layers[i].train_able = False


# In[6]:

from keras.optimizers import Adam


# ## create generator

# In[7]:

from model.generator import Generator
from keras.callbacks import TensorBoard


# In[8]:

tensorboard = TensorBoard(r'E:\yolov3\tf')


# In[ ]:




# In[9]:

from keras.utils import multi_gpu_model
parallel_model = multi_gpu_model(model)
# complie
parallel_model.compile(optimizer=Adam(lr=1e-3), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: K.mean(y_pred)})


# In[10]:

batch_size = 24
gen = Generator(r'C:\dataset\jinnan2_round1_train_20190305\jinnan2_round1_train_20190305\restricted\train.txt',
                batch_size = batch_size,
                num_classes = config.num_classes,
                shape  = (416,416))


# In[11]:

model.load_weights(r'stage1.hdf5')
res = parallel_model.fit_generator(gen,
                          steps_per_epoch =gen.num_samples()// batch_size,
                          epochs = 50,
                          max_queue_size=128,
                          workers=4,
                          verbose=1)


# In[12]:

model.save_weights(r'stage1.hdf5')


# In[ ]:

n = [print(n.name) for n in tf.get_default_graph().as_graph_def().node]


# In[ ]:



