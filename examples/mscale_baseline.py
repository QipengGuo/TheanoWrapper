import numpy as NP
import theano
import theano.tensor as T
import theano.tensor.nnet as NN
from wrapper import *
from ptb_data import *
fname = 'ptb_baseline'
ptb = ptb_data()
in_dim = ptb.c_dim
out_dim = ptb.c_dim
gru1_dim = 200
train_batch_size = 32
test_batch_size = 32
mask_value = 1.0
char_in = T.tensor3()
char_target = T.tensor3()

model = Model()
def get_mask(cur_in, mask_value=0.):
    return T.shape_padright(T.any((1. - T.eq(cur_in, mask_value)), axis=-1))

def _step(char, trash, prev_h1):
	batch_mask = get_mask(char)
	gru1 = batch_mask * model.gru(cur_in = char, rec_in = prev_h1, name = 'gru1', shape = (in_dim, gru1_dim))
	fc1 = batch_mask * NN.softmax(model.fc(cur_in = gru1, name = 'fc1', shape = (gru1_dim, out_dim)))
	return fc1, gru1

sc, _ = theano.scan(_step, sequences=[char_in.dimshuffle([1, 0, 2])], outputs_info = [T.zeros((char_in.shape[0], out_dim)),T.zeros((char_in.shape[0], gru1_dim))])

char_out = sc[0].dimshuffle([1, 0, 2])
EPSI = 1e-6
cost = T.mean(NN.categorical_crossentropy(T.clip(char_out, EPSI, 1.0-EPSI), char_target))
grad = rmsprop(cost, model.weightsPack.getW_list(), lr=1e-2, epsilon=1e-4)
train_func = theano.function([char_in, char_target], [cost, char_out], updates=grad, allow_input_downcast=True)
test_func = theano.function([char_in, char_target], [cost, char_out], allow_input_downcast=True)

test_his = []
for i in xrange(100):
    for j in xrange(500):
        data, label, mask  = ptb.get_batch(train_batch_size)
	data = data[:, :-1]
	label = label[:, 1:]
        n_cost, net_out = train_func(data, label)
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==NP.argmax(label, 2)).astype('int')+(NP.any(label>0, axis=-1)).astype('int')>1)/NP.sum(NP.any(label>0, axis=-1))
        per = NP.sum(-1.0*label*NP.log(net_out+EPSI))/NP.sum(label>0)
        print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' ACC = ', acc, ' Per = ', per
 
    test_acc = []
    test_per = []
    for j in xrange(ptb.test_size/test_batch_size):
        data, label, mask = ptb.get_batch(test_batch_size, test=True)
	data = data[:, :-1]
	label = label[:, 1:]
        n_cost, net_out = test_func(data, label)
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==NP.argmax(label, 2)).astype('int')+(NP.any(label>0, axis=-1)).astype('int')>1)/NP.sum(NP.any(label>0, axis=-1))
        per = NP.sum(-1.0*label*NP.log(net_out+EPSI))/NP.sum(label>0)
        print ' Test Batch = ', j,  
        test_acc.append(acc)
        test_per.append(per)
    print '\nEpoch = ', str(i), ' Test Acc = ', NP.mean(NP.asarray(test_acc)), ' Test Per = ', NP.mean(NP.asarray(test_per))
    NP.savez(fname+'_test_show.npz', label=label, predict=net_out)
    test_his.append(NP.mean(NP.asarray(test_per)))
    model.save(fname)
model.save(fname)
NP.savez(fname+'_result.npz', test_his = test_his)

