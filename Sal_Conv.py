import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import numpy as NP
import numpy.random as RNG
import h5py
import dataUtils
from collections import OrderedDict
from Wrapper import Model, rmsprop,  sgd
from theano.tensor.signal import downsample
from getopt import *
import sys

fname = 'sal_conv'
train_batch_size = 64
test_batch_size = 64
test_his = []

train_num_digits = 1
test_num_digits = 1
with_clutters = False


try:
	opts, args = getopt(sys.argv[1:], "", ["with_clutters", "train_num=", "test_num="])
	for opt in opts:
		if opt[0] == "--with_clutters":
			with_clutters = True
                elif opt[0] == "--train_num":
			train_num_digits = int(opt[1])
		elif opt[0] == "--test_num":
			test_num_digits = int(opt[1])
	if len(args) > 0:
		fname = args[0]
except:
	pass


print 'train_num_digits=',train_num_digits
print 'test_num_digits=',test_num_digits
print 'file_name=',fname
print 'with_clutters=',with_clutters

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
    #if train_flag:
    #    conv1_act = model.dropout(cur_in = conv1_act, name = 'dropout1', shape=(1, 1), prob=0.5)
    
    conv2_shape = (50, 20, 5, 5, 1, 1)
    conv2_h = model.conv(cur_in = conv1_act, name = 'conv2', shape=conv2_shape)
    pool2 = downsample.max_pool_2d(conv2_h, maxpool_shape, ignore_border = True)
    conv2_act = T.tanh(pool2)
    #if train_flag:
    #    conv1_act = model.dropout(cur_in = conv1_act, name = 'dropout2', shape=(1, 1), prob=0.5)
    
    conv3_shape = (50, 50, 5, 5, 1, 1)
    conv3_h = model.conv(cur_in = conv2_act, name = 'conv3', shape=conv3_shape)
    pool3 = downsample.max_pool_2d(conv3_h, maxpool_shape, ignore_border = True)
    conv3_act = T.tanh(pool3)
    if train_flag:
        conv3_act = model.dropout(cur_in = conv3_act, name = 'dropout3', shape=(1, 1), prob=0.25)
    
    fc1 = NN.sigmoid(model.fc(cur_in = conv3_act.flatten(2), name = 'fc1', shape=(4050, out_dim)))
    #fc1 = NN.softmax(model.fc(cur_in = conv2_act.flatten(2), name = 'fc1', shape=(24200, out_dim)))
    return fc1


_EPSI = 1e-6
MAX_V = 20

train_flag=True
img_out = _step(img_in)
cost = T.mean(T.sum(NN.binary_crossentropy(T.clip(img_out, _EPSI, 1.0 - _EPSI), img_tar), axis=1))
#cost = T.mean(NN.categorical_crossentropy(T.clip(img_out, _EPSI, 1.0 - _EPSI), img_tar))
cost_sum_sal = 0.0
cost_sal = [None]*10
for i in xrange(10):
    sal_img = T.grad(T.sum(img_out[:,i]), img_in)
    sal_img = sal_img / (T.sum(sal_img)+_EPSI)
    cost_sal[i] = T.sum((sal_img-seg_img[:,i:i+1])**2)
    cost_sal[i] = T.switch(cost_sal[i]>MAX_V, MAX_V, cost_sal[i])
    cost_sum_sal += cost_sal[i]
    
lr = theano.shared(NP.array((1e-3), dtype=NP.float32))
iterations = theano.shared(NP.array((0.), dtype=NP.float32))

train_func = theano.function([img_in, img_tar, seg_img], [cost, img_out], updates=rmsprop(cost+cost_sum_sal, model.weightsPack.getW_list()), allow_input_downcast=True)


train_flag=False
img_out = _step(img_in)
cost = T.mean(T.sum(NN.binary_crossentropy(T.clip(img_out, _EPSI, 1.0 - _EPSI), img_tar), axis=1))
#cost = T.mean(NN.categorical_crossentropy(T.clip(img_out, _EPSI, 1.0 - _EPSI), img_tar))
cost_sum_sal = 0.0
cost_sal = [None]*10
for i in xrange(10):
    sal_img = T.grad(T.sum(img_out[:,i]), img_in)
    sal_img = sal_img / (T.sum(sal_img)+_EPSI)
    cost_sal[i] = T.sum((sal_img-seg_img[:,i:i+1])**2)
    cost_sal[i] = T.switch(cost_sal[i]>MAX_V, MAX_V, cost_sal[i])
    cost_sum_sal += cost_sal[i]
    
test_func = theano.function([img_in, img_tar, seg_img], [cost, img_out, cost_sum_sal], allow_input_downcast=True)
#Network end


#training process
from dataUtils import BouncingMNIST
image_size = 100
bmnist = BouncingMNIST(train_batch_size, image_size, 'train/inputs', 'train/targets', with_clutters=with_clutters, clutter_size_max = 14, scale_range = 0.1)
bmnist_test = BouncingMNIST(test_batch_size, image_size, 'test/inputs', 'test/targets', with_clutters=with_clutters, clutter_size_max = 14, scale_range = 0.1)
for i in xrange(50):
    for j in xrange(1000):
        #data, label, seg=bmnist.GetStaticBatch(num_digits=RNG.randint(5))
        data, label, seg=bmnist.GetStaticBatch(num_digits=RNG.randint(train_num_digits)+1)
        n_cost, net_out = train_func(data, label, seg)
        #acc = NP.mean(NP.argmax(net_out, axis=1)==NP.argmax(label, axis=1))
        acc = 1.0 * NP.sum(net_out+label>1.5)/NP.sum(net_out+label>0.5)
        print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' ACC = ', acc
    test_acc = []
    test_sal_cost = []
    for j in xrange(len(bmnist_test.data_)/test_batch_size):
        data, label, seg=bmnist_test.GetStaticBatch(num_digits=test_num_digits)
        n_cost, net_out, n_sal_cost = test_func(data, label, seg)
        test_sal_cost.append(n_sal_cost)
        test_acc.append(1.0*NP.sum(net_out+label>1.5)/NP.sum(net_out+label>0.5))
        #test_acc = NP.mean(NP.argmax(net_out, axis=1)==NP.argmax(label, axis=1))
    print 'Epoch = ', str(i), ' Batch = ', str(j), ' Test ACC = ', NP.mean(NP.asarray(test_acc)), 'Test Sal Cost = ', NP.mean(NP.asarray(test_sal_cost))
    test_his.append(NP.mean(NP.asarray(test_acc)))
    model.save(fname)

model.save(fname)
NP.savez(fname+'_result.npz', test_his = test_his)
