# TheanoWrapper
Try to make some useful wrappers of theano for RNN 

# Need theano git version, not 0.7.0 release

#How to use 
```python

#these package are necessary in common
import theano 
import theano.tensor as T
import theano.tensor.nnet as NN
import numpy as NP

from wrapper import * #import tools

#prepare your data in numpy format 


#declare input and output variable
in_data = T.matrix() #a simple mlp network, two axis are (samples, channels)
label = T.matrix()

model = Model #declare a model 

#define network working flow
def _step(X):
    fc1 = T.tanh(model.fc(cur_in = X, name='fc1', shape=((in_dim, h_dim)))) #fc means fully connect layer
    fc2 = T.tanh(model.fc(cur_in = fc1, name='fc2', shape=((h_dim, out_dim)))) 
    return fc2

net_out = _step(in_data) #get the output of network

loss = T.mean((net_out-label)**2) #define loss function

rms = rmsprop(cost, model.weightsPack.getW_list(), lr=1e-2, epsilon=1e-4) #define optimizer, getW_list() return all parameters, don't decalre unused layers in model 
train_func = theano.function([in_data, label], [loss, net_out], updates=rms, allow_input_downcast=True)
test_func = theano.function([in_data, label], [loss, net_out], allow_input_downcast=True)

#one epoch, seen each training sample once
train_cost, _ = auto_batch(train_func, train_batch_size, train_data, train_label) # train_data, train_label is the real data
test_cost, _ = auto_batch(test_func, test_batch_size, test_data, test_label)

model.save('abc') #save the model in npz format, the network can't share saved file on cpu and gpu at now
model.load('abc')

```

# Some Examples
```python
#this code show a example for word-level language model on amazon reviews dataset
import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import numpy as NP
from apr_data import * #data parser
from wrapper import * #import our tools

#setting
fname = 'apr_W' #path to store model and history
train_batch_size = 16
test_batch_size = 32
test_his = []

#init data parser
amazon_data = Amazon_data('', '', cache=True) #using exist cache

#model design
word_seq = T.tensor3()
label_seq = T.tensor3()
in_dim = amazon_data.vocab_size
out_dim = amazon_data.vocab_size
starts = T.matrix() #be all zero in this experiments
model = Model()

#model.load(fname) load the stored weights

#You need a mask for variance input length for batch   
def get_mask(cur_in, mask_value=0.):
	return T.shape_padright(T.any((1. - T.eq(cur_in, mask_value)), axis=-1))

def _step(cur_in, trash, prev_h1, prev_h2, prev_h3):  #cur_in -- current input, trash -- the prev final output uesless in this experiments, prev_h1, prev_h2, prev_h3 -- prev hidden state for GRU
	mask = get_mask(cur_in)
	gru1 = mask * model.gru(cur_in = cur_in, rec_in = prev_h1, name = 'gru1', shape=(in_dim, 200))
	gru2 = mask * model.gru(cur_in = gru1, rec_in = prev_h2, name = 'gru2', shape=(200, 200))
	gru3 = mask * model.gru(cur_in = gru2, rec_in = prev_h3, name = 'gru3', shape=(200,200))
	#these design a three layer gru network, the same name layer will share the weights
	
	fc1 = mask * NN.softmax(model.fc(cur_in = gru3, name = 'fc1', shape =(200, out_dim)))
	
	return fc1, gru1, gru2, gru3 #be the next trash, prev_h1, prev_h2, prev_h3
	
_word_seq = word_seq.dimshuffle(1, 0, 2) #shuffle the time axis to the first
sc, _ = theano.scan(_step, sequences=[_word_seq], outputs_info=[starts, T.zeros((word_seq.shape[0], 200)), T.zeros((word_seq.shape[0], 200)), T.zeros((word_seq.shape[0], 200))], truncate_gradient=-1) #truncate_gradient be k(k>0), means bp through k time steps
word_out = sc[0].dimshuffle(1, 0, 2) #recover the axis to be (batch_size, time, channel)

EPSI = 1e-6
cost = T.sum(NN.categorical_crossentropy(T.clip(word_out, EPSI, 1.0-EPSI), label_seq))
test_func = theano.function([word_seq, label_seq, starts], [cost, word_out], allow_input_downcast=True)
train_func = theano.function([word_seq, label_seq, starts], [cost, word_out], updates=rmsprop(cost, model.weightsPack.getW_list(), lr=1e-2, epsilon=1e-4), allow_input_downcast=True)
#the whole network has compiled into two function, test and train


#training process
for i in xrange(50):
    for j in xrange(200):
        data, label, n_len = amazon_data.get_batch(train_batch_size)
        n_cost, net_out = train_func(data, label, NP.zeros((train_batch_size, len(data[0][0]))))
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==NP.argmax(label, 2)).astype('int')+(NP.any(label>0, axis=-1)).astype('int')>1)/NP.sum(NP.any(label>0, axis=-1))
        per = 1.0 * NP.exp(NP.mean(NP.sum(-1.0*label*NP.log(net_out+EPSI), axis=(1, 2))/n_len))
        print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' ACC = ', acc, ' Per = ', per, ' Sent Len = ', NP.mean(n_len)
 
    test_acc = []
    test_per = []
    for j in xrange(amazon_data.test_size/test_batch_size):
        data, label, n_len = amazon_data.get_batch(test_batch_size, test=True)
        n_cost, net_out = test_func(data, label, NP.zeros((test_batch_size, len(data[0][0]))))
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==NP.argmax(label, 2)).astype('int')+(NP.any(label>0, axis=-1)).astype('int')>1)/NP.sum(NP.any(label>0, axis=-1))
        per = 1.0 * NP>exp(NP.mean(NP.sum(-1.0*label*NP.log(net_out+EPSI), axis=(1, 2))/n_len))
        print ' Test Batch = ', j,  
        test_acc.append(acc)
        test_per.append(per)
    print '\nEpoch = ', str(i), ' Test Acc = ', NP.mean(NP.asarray(test_acc)), ' Test Per = ', NP.mean(NP.asarray(test_per))
    NP.savez('apr_test_show.npz', label=label, predict=net_out)
    test_his.append(NP.mean(NP.asarray(test_per)))
    model.save(fname)
model.save(fname)
NP.savez(fname+'_result.npz', test_his = test_his)
```

#Conv network
```python
...
from Wrapper import Model, rmsprop
from theano.tensor.signal import downsample

...
#Mnist handwriting recognition 
#Network design
img_shape = (100, 100)
out_dim = 10
model = Model()
img_in = T.tensor4()
seg_img = T.tensor4()
img_tar = T.matrix()

def _step(cur_in):
    
    maxpool_shape=(2, 2)
    
    conv1_shape = (20, 1, 5, 5, 1, 1)
    conv1_h = model.conv(cur_in = cur_in, name = 'conv1', shape=conv1_shape)
    pool1 = downsample.max_pool_2d(conv1_h, maxpool_shape, ignore_border = True)
    conv1_act = T.tanh(pool1)
    
    conv2_shape = (50, 20, 5, 5, 1, 1)
    conv2_h = model.conv(cur_in = conv1_act, name = 'conv2', shape=conv2_shape)
    pool2 = downsample.max_pool_2d(conv2_h, maxpool_shape, ignore_border = True)
    conv2_act = T.tanh(pool2)
    
    conv3_shape = (50, 50, 5, 5, 1, 1)
    conv3_h = model.conv(cur_in = conv2_act, name = 'conv3', shape=conv3_shape)
    pool3 = downsample.max_pool_2d(conv3_h, maxpool_shape, ignore_border = True)
    conv3_act = T.tanh(pool3)
    if train_flag:
        conv3_act = model.dropout(cur_in = conv3_act, name = 'dropout3', shape=(1, 1), prob=0.25)
    
    fc1 = NN.sigmoid(model.fc(cur_in = conv3_act.flatten(2), name = 'fc1', shape=(4050, 128)))
    fc2 = NN.softmax(model.fc(cur_in = fc1, name='fc2', shape=(128,out_dim)))
    
    return fc2


_EPSI = 1e-6
MAX_V = 20

train_flag=True
img_out = _step(img_in)
cost = T.mean(NN.categorical_crossentropy(T.clip(img_out, _EPSI, 1.0 - _EPSI), img_tar))

train_func = theano.function([img_in, img_tar], [cost, img_out], updates=rmsprop(cost, model.weightsPack.getW_list()), allow_input_downcast=True)


train_flag=False
img_out = _step(img_in)
cost = T.mean(NN.categorical_crossentropy(T.clip(img_out, _EPSI, 1.0 - _EPSI), img_tar)) 

test_func = theano.function([img_in, img_tar], [cost, img_out], allow_input_downcast=True)
#Network end
```
