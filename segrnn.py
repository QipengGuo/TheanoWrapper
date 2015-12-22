import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import numpy as NP
from apr_data import *
from Wrapper import Model, rmsprop
from collections import OrderedDict
import sys
fname = 'att_rnn'
train_batch_size = 16
test_batch_size = 16
test_his = []

#opt here

amazon_data = Amazon_data('item.json.gz', 'wordsEn.txt', cache=True)

word_seq = T.tensor3()
label_seq = T.tensor3()
in_dim = amazon_data.vocab_size
out_dim = amazon_data.vocab_size
starts = T.matrix()
time_step = theano.shared(NP.zeros([], dtype='int'))
#time_step = T.zeros([], dtype='int')

max_time = 199
#mem_bank time, batch, channels
mem_bank = theano.shared(NP.zeros((max_time, train_batch_size, 256), dtype=theano.config.floatX))
#mem_bank = T.zeros((1,word_seq.shape[0],200))
model = Model()

#merge two ordered dict
def merge_OD(A, B):
    C=OrderedDict()
    for k,e in A.items()+B.items():
        C[k]=e
    return C

def batched_dot(A, B):
    #borrowed from Draw, Google Deep mind group
    C = A.dimshuffle([0, 1, 2, 'x']) * B.dimshuffle([0, 'x', 1, 2])
    return C.sum(axis=-2)
                
def get_mask(cur_in, mask_value=0.):
    return T.shape_padright(T.any((1. - T.eq(cur_in, mask_value)), axis=-1))

def _step_rnn(st, ed, prev_h1, mem, inputs): 
    mask = get_mask(inputs[ed])
    gru1 = mask * model.gru(cur_in = inputs[ed], rec_in = T.switch(st==ed, T.zero_like(prev_h1), prev_h1), name = 'gru1', shape = (in_dim, mem_dim))
    mem = T.set_subtensor(mem[st][ed], gru1)
    return gru1, mem

def _step_seg(st, ed, trash1, trash2, mem, match_ref):
    mask = get_mask(mem[st][ed])
    fc1 = model.fc(cur_in = mem[st][ed], name = 'fc1', shape=(mem_dim, 200))
    fc2 = T.tanh(model.fc(cur_in = fc1, name = 'fc2', shape=(200, 1)))
    fc2 = T.extra_ops.squeeze(T.patternbroadcast(fc2, (False, False, True)))
    return fc2, fc2 * match_ref[st][ed]


# TO DO ...



_word_seq = word_seq.dimshuffle(1, 0, 2)
sc, _ = theano.scan(_step, sequences=[_word_seq, T.arange(_word_seq.shape[0])], outputs_info=[T.zeros((word_seq.shape[0], int(out_dim))), mem_bank])
#sc, sc_updates = theano.scan(_step, sequences=[_word_seq], outputs_info=[starts])
word_out = sc[0].dimshuffle(1, 0, 2)

EPSI = 1e-6
cost = T.sum(NN.categorical_crossentropy(T.clip(word_out, EPSI, 1.0-EPSI), label_seq))
#test_func = theano.function([word_seq, label_seq, starts], [cost, word_out], updates=sc_updates, allow_input_downcast=True)
test_func = theano.function([word_seq, label_seq], [cost, word_out], allow_input_downcast=True)
grad = rmsprop(cost, model.weightsPack.getW_list(), lr=1e-2, epsilon=1e-4)
#train_func = theano.function([word_seq, label_seq, starts], [cost, word_out], updates=merge_OD(sc_updates,grad), allow_input_downcast=True)
train_func = theano.function([word_seq, label_seq], [cost, word_out], updates=grad, allow_input_downcast=True)


for i in xrange(50):
    if i == 0:
        data, label, n_len = amazon_data.get_batch(test_batch_size)
        print NP.shape(data)
        print NP.shape(label)
        n_cost, net_out = test_func(data, label)
        print NP.shape(net_out)
        NP.savez(fname+'_first_show.npz', label=label, predict=net_out)
        #sys.exit()
    for j in xrange(200):
        data, label, n_len = amazon_data.get_batch(train_batch_size)
        #data = NP.zeros((train_batch_size, len(data[0]), len(data[0][0])))
        n_cost, net_out = train_func(data, label)
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==NP.argmax(label, 2)).astype('int')+(NP.any(label>0, axis=-1)).astype('int')>1)/NP.sum(NP.any(label>0, axis=-1))
        per = 1.0 * NP.mean(NP.sum(-1.0*label*NP.log(net_out+EPSI), axis=(1, 2))/n_len)
        print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' ACC = ', acc, ' Per = ', per, ' Sent Len = ', NP.mean(n_len)
 
    test_acc = []
    test_per = []
    for j in xrange(amazon_data.test_size/test_batch_size):
        data, label, n_len = amazon_data.get_batch(test_batch_size, test=True)
        n_cost, net_out = test_func(data, label)
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==NP.argmax(label, 2)).astype('int')+(NP.any(label>0, axis=-1)).astype('int')>1)/NP.sum(NP.any(label>0, axis=-1))
        per = 1.0 * NP.mean(NP.sum(-1.0*label*NP.log(net_out+EPSI), axis=(1, 2))/n_len)
        print ' Test Batch = ', j,  
        test_acc.append(acc)
        test_per.append(per)
    print '\nEpoch = ', str(i), ' Test Acc = ', NP.mean(NP.asarray(test_acc)), ' Test Per = ', NP.mean(NP.asarray(test_per))
    NP.savez(fname+'_test_show.npz', label=label, predict=net_out)
    test_his.append(NP.mean(NP.asarray(test_per)))
    model.save(fname)
model.save(fname)
NP.savez(fname+'_result.npz', test_his = test_his)



