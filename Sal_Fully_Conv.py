import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import numpy as NP
import numpy.random as RNG
import h5py
import dataUtils
from collections import OrderedDict
from Wrapper import Model, rmsprop
%matplotlib inline
import matplotlib.pyplot as plt
from theano.tensor.signal import downsample


def get_conv_out_dim(img_shape, conv_shape):
    img_row, img_col = img_shape
    nr_filters, nr_channels, filter_row, filter_col, stride_row, stride_col = conv_shape
    output_dim = ((img_row - filter_row) / stride_row + 1) * ((img_col - filter_col) / stride_col + 1) * nr_filters * 1
    return output_dim

def get_conv_out_shape(img_shape, conv_shape):
    img_row, img_col = img_shape
    nr_filters, nr_channels, filter_row, filter_col, stride_row, stride_col = conv_shape
    return (((img_row - filter_row) / stride_row + 1), ((img_col - filter_col) / stride_col + 1))

def _step(cur_in):
    
    conv1_shape = (20, 1, 5, 5, 2, 2)
    
    conv1_h = model.conv(cur_in = cur_in, name = 'conv1', shape=conv1_shape)
    conv1_out_shape = get_conv_out_shape(img_shape, conv1_shape)
    conv1_h = T.tanh(conv1_h)
    
    conv2_shape = (50, 20, 5, 5, 2, 2)
    conv2_h = model.conv(cur_in = conv1_h, name = 'conv2', shape=conv2_shape)
    conv2_h = T.tanh(conv2_h)
    
    fc1 = NN.sigmoid(model.fc(cur_in = conv2_h.flatten(2), name = 'fc1', shape=(24200, 10)))
    #fc2 = NN.softmax(model.fc(cur_in = fc1, name = 'fc2', shape=(100, out_dim)))
    return fc1
    

img_shape = (100, 100)
out_dim = 10
model = Model()
img_in = T.tensor4()
seg_img = T.tensor4()
img_tar = T.matrix()
img_out = _step(img_in)
_EPSI = 1e-6
cost = T.mean(T.sum(NN.binary_crossentropy(T.clip(img_out, _EPSI, 1.0 - _EPSI), img_tar), axis=1))
#cost = T.mean(NN.categorical_crossentropy(T.clip(img_out, _EPSI, 1.0 - _EPSI), img_tar))
MAX_V = 20
cost_sum_sal = 0.0
cost_sal = [None]*10
for i in xrange(10):
    sal_img = T.grad(T.sum(img_out[:,i]), img_in)
    sal_img = sal_img / (T.sum(sal_img)+_EPSI)
    cost_sal[i] = T.sum((sal_img-seg_img[:,i:i+1])**2)
    cost_sal[i] = T.switch(cost_sal[i]>MAX_V, MAX_V, cost_sal[i])
    cost_sum_sal += cost_sal[i]
    
#grad_func = theano.function([img_in, img_tar], rms_grad.values()[4], allow_input_downcast=True)    
#test_func = theano.function([img_in, img_tar, seg_img], cost, allow_input_downcast=True)
test_sal_fc_func = theano.function([img_in, img_tar, seg_img], [cost, img_out, cost_sum_sal], allow_input_downcast=True)
train_sal_fc_func = theano.function([img_in, img_tar, seg_img], [cost, img_out, cost_sum_sal], updates=rmsprop(cost+cost_sum_sal, model.weightsPack.getW_list(), lr=1e-4, rho=0.9, epsilon=1e-8), allow_input_downcast=True)
model_sal_fc = model
#predict_func = theano.function([img_in], img_out, allow_input_downcast=True)

train_batch_size = 64
test_batch_size = 64
test_sal_fc_his = []

from dataUtils import BouncingMNIST
image_size = 100
bmnist = BouncingMNIST(train_batch_size, image_size, 'train/inputs', 'train/targets', with_clutters=False, clutter_size_max = 14, scale_range = 0.1)
bmnist_test = BouncingMNIST(test_batch_size, image_size, 'test/inputs', 'test/targets', with_clutters=False, clutter_size_max = 14, scale_range = 0.1)
for i in xrange(50):
    for j in xrange(200):
        #data, label, seg=bmnist.GetStaticBatch(num_digits=RNG.randint(5))
        data, label, seg=bmnist.GetStaticBatch(num_digits=1)
        n_cost, net_out, n_cost_sal = train_sal_fc_func(data, label, seg)
        #acc = NP.mean(NP.argmax(net_out, axis=1)==NP.argmax(label, axis=1))
        acc = NP.sum((net_out+label>1.5))/NP.sum((net_out+label>0.5))
        print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' Cost Sal = ', n_cost_sal, ' ACC = ', acc
    
    test_acc = []
    test_sal_cost = []
    for j in xrange(len(bmnist_test.data_)/test_batch_size):
        data, label, seg=bmnist_test.GetStaticBatch(num_digits=1)
        n_cost, net_out, n_sal_cost = test_sal_fc_func(data, label, seg)
        test_sal_cost.append(n_sal_cost)
        test_acc.append(NP.mean(NP.sum(((net_out>0.5)==(label>0.5))==(label>0.5), axis=1)))
        #test_acc = NP.mean(NP.argmax(net_out, axis=1)==NP.argmax(label, axis=1))
    print 'Epoch = ', str(i), ' Batch = ', str(j), ' Test ACC = ', NP.mean(NP.asarray(test_acc)), 'Test Sal Cost = ', NP.mean(NP.asarray(test_sal_cost))
    test_sal_fc_his.append(NP.mean(NP.asarray(test_acc)))

model_sal_fc.save('sal_fully_conv')
NP.savez('sal_fully_conv_result.npz', test_his = test_sal_fc_his)