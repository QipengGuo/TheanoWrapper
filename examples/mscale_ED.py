import numpy as NP
import theano
import theano.tensor as T
import theano.tensor.nnet as NN
from wrapper import *
from ptb_data import *

ptb = ptb_data()
in_dim = ptb.c_dim
out_dim = ptb.c_dim
fname = 'ptb_ED'
train_batch_size = 32
test_batch_size = 32
mask_value = 1.0
char_in = T.tensor3()
cw_mask = T.matrix() # char to word
gru1_dim = 200
gru2_dim = 200
mem_dim = gru1_dim
char_target = T.tensor3()

model = Model()
def get_mask(cur_in, mask_value=0.):
    return T.shape_padright(T.any((1. - T.eq(cur_in, mask_value)), axis=-1))

#a b c I J K    input
#0 0 1 0 0 1    mask
def encode(char, mask, prev_h1, mem):
        batch_mask = get_mask(char)
	gru1 =  model.gru(cur_in = char, rec_in = prev_h1, name = 'gru_enc', shape = (in_dim, gru1_dim))
	mask = T.shape_padright(mask)
        mem = batch_mask * (mask * gru1 + (1-mask) * mem)
        next_gru1 = batch_mask * mask * gru1
	return next_gru1, mem

#NULL a b c I J    input
#a    b c I J K    output
#0    0 0 1 0 0 1  mask

#NULL  K J I c b
#K     J I c b a
#1     0 0 1 0 0 1
def decode(enc, mask, pred_char, prev_h1, word_emb):
        batch_mask = get_mask(enc)
        mask = T.shape_padright(mask)
	prev_h1 = prev_h1 * (1-mask)
	pred_char = pred_char * (1-mask)
        word_emb = mask * enc + (1-mask) * word_emb
	gru1 = batch_mask * model.gru(cur_in = T.concatenate((pred_char, word_emb), axis=1), rec_in = prev_h1, name = 'gru_dec', shape = (out_dim+gru1_dim, gru2_dim))
	fc1 = batch_mask * NN.softmax(model.fc(cur_in = gru1, name = 'fc_dec', shape = (gru2_dim, out_dim)))
	return fc1, gru1, word_emb

#mem = theano.shared(NP.zeros((max_time, batch_size, mem_dim), dtype=theano.config.floatX))
#mem_idx = T.zeros((), dtype='int32')
mem_idx = theano.shared(NP.zeros((1), dtype='int'))
sc, _ = theano.scan(encode, sequences=[char_in.dimshuffle([1, 0, 2]), cw_mask.dimshuffle([1,0])], outputs_info =[T.zeros((char_in.shape[0], gru1_dim)), T.zeros((char_in.shape[0], gru1_dim))])

#time, batch, channel
enc = sc[-1][::-1]
rev_mask = cw_mask[:, ::-1]
sc,_ = theano.scan(decode, sequences=[enc, T.concatenate((T.ones((char_in.shape[0], 1)), rev_mask[:,:-1]), axis=1).dimshuffle([1, 0])], outputs_info = [T.zeros((char_in.shape[0], out_dim)), T.zeros((char_in.shape[0], gru2_dim)), T.zeros((char_in.shape[0], gru1_dim))])

char_out = (sc[0][::-1]).dimshuffle([1, 0, 2])

EPSI = 1e-6
cost = T.mean(NN.categorical_crossentropy(T.clip(char_out, EPSI, 1.0-EPSI), char_target))
grad = rmsprop(cost, model.weightsPack.getW_list(), lr=1e-2, epsilon=1e-4)
train_func = theano.function([char_in, cw_mask, char_target], [cost, char_out], updates=grad, allow_input_downcast=True)
test_func = theano.function([char_in, cw_mask, char_target], [cost, char_out], allow_input_downcast=True)

test_his = []
for i in xrange(100):
    for j in xrange(500):
        data, label, mask  = ptb.get_batch(train_batch_size)
	#print mask
        n_cost, net_out = train_func(data, mask, label)
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==NP.argmax(label, 2)).astype('int')+(NP.any(label>0, axis=-1)).astype('int')>1)/NP.sum(NP.any(label>0, axis=-1))
        per = NP.sum(-1.0*label*NP.log(net_out+EPSI))/NP.sum(label>0)
        print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' ACC = ', acc, ' Per = ', per
 
    test_acc = []
    test_per = []
    for j in xrange(ptb.test_size/test_batch_size):
        data, label, mask = ptb.get_batch(test_batch_size, test=True)
        n_cost, net_out = test_func(data, mask, label)
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==NP.argmax(label, 2)).astype('int')+(NP.any(label>0, axis=-1)).astype('int')>1)/NP.sum(NP.any(label>0, axis=-1))
        per = NP.sum(-1.0*label*NP.log(net_out+EPSI))/NP.sum(label>0)
        #print ' Test Batch = ', j,  
        test_acc.append(acc)
        test_per.append(per)
    print '\nEpoch = ', str(i), ' Test Acc = ', NP.mean(NP.asarray(test_acc)), ' Test Per = ', NP.mean(NP.asarray(test_per))
    NP.savez(fname+'_test_show.npz', label=label, predict=net_out)
    test_his.append(NP.mean(NP.asarray(test_per)))
    model.save(fname)
model.save(fname)
NP.savez(fname+'_result.npz', test_his = test_his)

