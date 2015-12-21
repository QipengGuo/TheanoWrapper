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

max_time = 250
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

def _step(cur_in, trash):
        updates=OrderedDict()
        updates[time_step] = (time_step+1) % max_time
        #updates.append((time_step, time_step+1))
        mask = get_mask(cur_in)
        fc1 = mask * T.tanh(model.fc(cur_in = cur_in, name = 'fc1', shape =(in_dim, 200)))
        att = model.att_mem(cur_in = fc1, mem_in = mem_bank, name = 'att1', shape = (200, 256), tick=time_step)

        rec_in1 = batched_dot(mem_bank[:time_step+1].dimshuffle([1, 2, 0]), att.dimshuffle([1, 0, 'x']))
        rec_in1 = T.extra_ops.squeeze(T.patternbroadcast(rec_in1, (False, False, True)))
        gru1 = mask * model.gru(cur_in = fc1, rec_in = rec_in1, name = 'gru1', shape = (200, 256))
        #gru2 = masked(model.gru(cur_in = gru1, rec_in = prev_h2, name = 'gru2', shape = (200, 200)), mask)
        #gru3 = masked(model.gru(cur_in = gru2, rec_in = prev_h3, name = 'gru3', shape = (200, 200)), mask)

        fc2 = mask * NN.softmax(model.fc(cur_in = gru1, name = 'fc2', shape = (256, out_dim)))

        new_mem_bank = T.set_subtensor(mem_bank[time_step], gru1)
        #new_mem_bank = T.concatenate((mem_bank, gru1.dimshuffle(['x', 0, 1])), axis=0)
        #updates.append((mem_bank, new_mem_bank))
        updates[mem_bank]=new_mem_bank
        return fc2, updates

_word_seq = word_seq.dimshuffle(1, 0, 2)
#sc, _ = theano.scan(_step, sequences=[_word_seq], outputs_info=[starts, T.zeros((word_seq.shape[0], 200)), T.zeros((word_seq.shape[0], 200)), T.zeros((word_seq.shape[0], 200))], truncate_gradient=200)
sc, sc_updates = theano.scan(_step, sequences=[_word_seq], outputs_info=[starts])
word_out = sc.dimshuffle(1, 0, 2)

EPSI = 1e-6
cost = T.sum(NN.categorical_crossentropy(T.clip(word_out, EPSI, 1.0-EPSI), label_seq))
test_func = theano.function([word_seq, label_seq, starts], [cost, word_out], updates=sc_updates, allow_input_downcast=True)
grad = rmsprop(cost, model.weightsPack.getW_list(), lr=1e-2, epsilon=1e-4)
train_func = theano.function([word_seq, label_seq, starts], [cost, word_out], updates=merge_OD(sc_updates,rmsprop(cost, model.weightsPack.getW_list(), lr=1e-2, epsilon=1e-4)), allow_input_downcast=True)
#train_func = theano.function([word_seq, label_seq, starts], [cost, word_out], updates=merge_OD(grad, OrderedDict()), allow_input_downcast=True)

for i in xrange(50):
    if i == 0:
        data, label, n_len = amazon_data.get_batch(test_batch_size)
        print NP.shape(data)
        print NP.shape(label)
        n_cost, net_out = test_func(data, label, NP.zeros((train_batch_size, len(data[0][0]))))
        print NP.shape(net_out)
        NP.savez(fname+'_first_show.npz', label=label, predict=net_out)
        #sys.exit()
    for j in xrange(200):
        data, label, n_len = amazon_data.get_batch(train_batch_size)
        #data = NP.zeros((train_batch_size, len(data[0]), len(data[0][0])))
        n_cost, net_out = train_func(data, label, NP.zeros((train_batch_size, len(data[0][0]))))
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==NP.argmax(label, 2)).astype('int')+(NP.any(label>0, axis=-1)).astype('int')>1)/NP.sum(NP.any(label>0, axis=-1))
        per = 1.0 * NP.mean(NP.sum(-1.0*label*NP.log(net_out+EPSI), axis=(1, 2))/n_len)
        print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' ACC = ', acc, ' Per = ', per, ' Sent Len = ', NP.mean(n_len)
 
    test_acc = []
    test_per = []
    for j in xrange(amazon_data.test_size/test_batch_size):
        data, label, n_len = amazon_data.get_batch(test_batch_size, test=True)
        n_cost, net_out = test_func(data, label, NP.zeros((train_batch_size, len(data[0][0]))))
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



