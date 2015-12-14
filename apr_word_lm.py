import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import numpy as NP
import numpy.random as RNG
import h5py
from apr_data import *
from Wrapper import Model, rmsprop
from getopt import *
import sys

fname = 'apr_W'
train_batch_size = 16
test_batch_size = 32
test_his = []

#opt here

amazon_data = Amazon_data('item.json.gz', 'wordsEn.txt', cache=True)

word_seq = T.tensor3()
label_seq = T.tensor3()
in_dim = amazon_data.vocab_size
out_dim = amazon_data.vocab_size
starts = T.matrix()
model = Model()

def get_mask(cur_in, mask_value=0.):
    return T.shape_padright(T.any((1. - T.eq(cur_in, mask_value)), axis=-1))
def masked(cur_in, mask):
    return cur_in * mask
def _step(cur_in, trash, prev_h1):
    mask = get_mask(cur_in)
    gru1 = masked(model.gru(cur_in = cur_in, rec_in = prev_h1, name = 'gru1', shape = (in_dim, 256)), mask)
    #gru2 = masked(model.gru(cur_in = gru1, rec_in = prev_h2, name = 'gru2', shape = (200, 200)), mask)
    #gru3 = masked(model.gru(cur_in = gru2, rec_in = prev_h3, name = 'gru3', shape = (200, 200)), mask)

    fc1 = masked(NN.softmax(model.fc(cur_in = gru1, name = 'fc1', shape = (256, out_dim))), mask)

    return fc1, gru1
_word_seq = word_seq.dimshuffle(1, 0, 2)
#sc, _ = theano.scan(_step, sequences=[_word_seq], outputs_info=[starts, T.zeros((word_seq.shape[0], 200)), T.zeros((word_seq.shape[0], 200)), T.zeros((word_seq.shape[0], 200))], truncate_gradient=200)
sc, _ = theano.scan(_step, sequences=[_word_seq], outputs_info=[starts, T.zeros((word_seq.shape[0], 256))], truncate_gradient=200)
word_out = sc[0].dimshuffle(1, 0, 2)

EPSI = 1e-6
cost = T.mean(NN.categorical_crossentropy(T.clip(word_out, EPSI, 1.0-EPSI), label_seq))
test_func = theano.function([word_seq, label_seq, starts], [cost, word_out], allow_input_downcast=True)
train_func = theano.function([word_seq, label_seq, starts], [cost, word_out], updates=rmsprop(cost, model.weightsPack.getW_list(), lr=1e-2, epsilon=1e-4), allow_input_downcast=True)

for i in xrange(50):
    if i == 0:
        data, label, n_len = amazon_data.get_batch(train_batch_size)
        n_cost, net_out = test_func(data, label, NP.zeros((train_batch_size, len(data[0][0]))))
        NP.savez('apr_first_show.npz', label=label, predict=net_out)
    for j in xrange(200):
        data, label, n_len = amazon_data.get_batch(train_batch_size)
        #data = NP.zeros((train_batch_size, len(data[0]), len(data[0][0])))
        n_cost, net_out = train_func(data, label, NP.zeros((train_batch_size, len(data[0][0]))))
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==NP.argmax(label, 2)).astype('int')+(NP.any(label>0, axis=-1)).astype('int')>1)/NP.sum(NP.any(label>0, axis=-1))
        per = 1.0 * NP.mean(NP.sum(-1.0*label*NP.log(net_out+EPSI), axis=(1, 2))/n_len)
        print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' ACC = ', acc, ' Per = ', per, ' Sent Len = ', NP.mean(n_len)

    NP.savez('apr_train_show.npz', label=label, predict=net_out)

    test_acc = []
    test_per = []
    for j in xrange(amazon_data.test_size/test_batch_size):
        data, label, n_len = amazon_data.get_batch(test_batch_size, test=True)
        n_cost, net_out = test_func(data, label, NP.zeros((test_batch_size, len(data[0][0]))))
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==NP.argmax(label, 2)).astype('int')+(NP.any(label>0, axis=-1)).astype('int')>1)/NP.sum(NP.any(label>0, axis=-1))
        per = 1.0 * NP.mean(NP.sum(-1.0*label*NP.log(net_out+EPSI), axis=(1, 2))/n_len)
        print ' Test Batch = ', j,  
        test_acc.append(acc)
        test_per.append(per)
    print '\nEpoch = ', str(i), ' Test Acc = ', NP.mean(NP.asarray(test_acc)), ' Test Per = ', NP.mean(NP.asarray(test_per))
    test_his.append(NP.mean(NP.asarray(test_per)))
    model.save(fname)
model.save(fname)
NP.savez(fname+'_result.npz', test_his = test_his)



