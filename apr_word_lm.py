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
test_batch_size = 16
test_his = []

#opt here

amazon_data = Amazon_data('item.json.gz', 'wordsEn.txt', cache=True)

word_seq = T.tensor3()
label_seq = T.tensor3()
in_dim = amazon_data.vocab_size
out_dim = amazon_data.vocab_size
starts = T.matrix()
model = Model()
def _step(cur_in, trash, prev_h1, prev_h2, prev_h3):
    gru1 = model.gru(cur_in = cur_in, rec_in = prev_h1, name = 'gru1', shape = (in_dim, 200))
    gru2 = model.gru(cur_in = gru1, rec_in = prev_h2, name = 'gru2', shape = (200, 200))
    gru3 = model.gru(cur_in = gru2, rec_in = prev_h3, name = 'gru3', shape = (200, 200))

    fc1 = NN.softmax(model.fc(cur_in = gru3, name = 'fc1', shape = (200, out_dim)))

    return fc1, gru1, gru2, gru3
_word_seq = word_seq.dimshuffle(1, 0, 2)
sc, _ = theano.scan(_step, sequences=[_word_seq], outputs_info=[starts, T.zeros((word_seq.shape[0], 200)), T.zeros((word_seq.shape[0], 200)), T.zeros((word_seq.shape[0], 200))], truncate_gradient=200)
word_out = sc[0].dimshuffle(1, 0, 2)

EPSI = 1e-6
cost = T.mean(NN.categorical_crossentropy(T.clip(word_out, EPSI, 1.0-EPSI), label_seq))
test_func = theano.function([word_seq, label_seq, starts], [cost, word_out], allow_input_downcast=True)
train_func = theano.function([word_seq, label_seq, starts], [cost, word_out], updates=rmsprop(cost, model.weightsPack.getW_list()), allow_input_downcast=True)

for i in xrange(50):
    for j in xrange(1000):
        data, label = amazon_data.get_batch(train_batch_size)
        #data = NP.zeros((train_batch_size, len(data[0]), len(data[0][0])))
        n_cost, net_out = train_func(data, label, NP.zeros((train_batch_size, len(data[0][0]))))
        acc = 1.0 * NP.mean(NP.argmax(net_out, 2)==NP.argmax(label, 2))
        print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' ACC = ', acc

    test_acc = []
    for j in xrange(amazon_data.test_size/test_batch_size):
        data, label = amazon_data.get_batch(test_batch_size, test=True)
        n_cost, net_out = test_func(data, label, NP.zeros((train_batch_size, len(data[0][0]))))
        acc = 1.0 * NP.mean(NP.argmax(net_out, 2)==NP.argmax(label, 2))
        test_acc.append(acc)
    print 'Epoch = ', str(i), ' Test ACC = ', NP.mean(NP.asarray(test_acc))
    test_his.append(NP.mean(NP.asarray(test_acc)))
    model.save(fname)
model.save(fname)
NP.savez(fname+'_result.npz', test_his = test_his)



