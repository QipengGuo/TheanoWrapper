import numpy as NP
import theano
import theano.tensor as T
import theano.tensor.nnet as NN
from wrapper import *
from ptb_data_word import *
fname = 'ptb_mem_alone'
ptb = ptb_data()
in_dim = ptb.c_dim
out_dim = ptb.c_dim
MAX_TIME = ptb.MAX_TIME
gru1_dim = 200
gru2_dim = 200
word_out_dim = ptb.w_dim
mem_dim = 100
train_batch_size = 32
test_batch_size = 32
mask_value = 1.0
char_in = T.tensor3()
char_target = T.tensor3()
cw_mask = T.matrix()
word_target = T.tensor3()
model = Model()
#model.load(fname)
def get_mask(cur_in, mask_value=0.):
    return T.shape_padright(T.any((1. - T.eq(cur_in, mask_value)), axis=-1))

def _step(char, mask, trash, prev_h1, word_emb, h_before, mem):
	batch_mask = get_mask(char)
	gru1 = batch_mask * model.gru(cur_in = char, rec_in = prev_h1, name = 'gru1', shape = (in_dim, gru1_dim))
	att = model.att_mem(cur_in = gru1-h_before, mem_in = mem, name = 'att1', shape = (gru1_dim, mem_dim), tick=MAX_TIME)
        mask =T.shape_padright(mask)
	mem_in = T.sum(mem.dimshuffle([1, 0, 2]) * att.dimshuffle([1, 0, 'x']), axis=1)
	word_emb = mem_in * mask + word_emb * (1-mask)
        h_before = gru1 * mask + h_before * (1-mask)
        fc1 = batch_mask * NN.softmax(model.fc(cur_in = gru1, name = 'fc1', shape = (gru1_dim, out_dim)))

	return fc1, gru1, word_emb,h_before

def word_step(word_emb, mask, trash, prev_h1):
	batch_mask = get_mask(mask)
	gru1 = batch_mask * model.gru(cur_in = word_emb, rec_in = prev_h1, name = 'gru_word', shape = (mem_dim, word_out_dim))
	next_h1 = mask * gru1 + (1-mask) * prev_h1
	fc1 = batch_mask * NN.softmax(model.fc(cur_in = gru1, name = 'fc_word', shape = (gru2_dim, word_out_dim)))
	return fc1, next_h1

#NULL a b c I J    input
#a    b c I J K    output
#0    0 0 1 0 0 1  mask

#NULL  K J I c b
#K     J I c b a
#1     0 0 1 0 0 1
mem = model.Wmatrix(name='mem', shape=(MAX_TIME, mem_dim)).dimshuffle(0, 'x', 1)
sc, _ = theano.scan(_step, sequences=[char_in.dimshuffle([1, 0, 2]), cw_mask.dimshuffle([1, 0])], outputs_info = [T.zeros((char_in.shape[0], out_dim)),T.zeros((char_in.shape[0], gru1_dim)), T.zeros((char_in.shape[0], mem_dim)), T.zeros((char_in.shape[0], gru1_dim))], non_sequences=[mem])

char_out = sc[0].dimshuffle([1, 0, 2])
char_out = char_out[:, :-1]
EPSI = 1e-6
cost = T.mean(NN.categorical_crossentropy(T.clip(char_out, EPSI, 1.0-EPSI), char_target[:,1:]))

enc = sc[-2]
sc,_ = theano.scan(word_step, sequences=[enc, T.concatenate((T.ones((char_in.shape[0], 1)), cw_mask[:,:-1]), axis=1).dimshuffle([1, 0])], outputs_info = [T.zeros((char_in.shape[0], word_out_dim)), T.zeros((char_in.shape[0], gru2_dim))])

word_out = (sc[0]).dimshuffle([1, 0, 2])

cost += T.mean(NN.categorical_crossentropy(T.clip(word_out, EPSI, 1.0-EPSI), word_target))


grad = rmsprop(cost, model.weightsPack.getW_list(), lr=1e-2, epsilon=1e-4)
train_func = theano.function([char_in, cw_mask, char_target, word_target], [cost, char_out, word_out], updates=grad, allow_input_downcast=True)
test_func = theano.function([char_in, cw_mask, char_target, word_target], [cost, char_out, word_out], allow_input_downcast=True)

def evaluate(net_out, label):
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==NP.argmax(label, 2)).astype('int')+(NP.any(label>0, axis=-1)).astype('int')>1)/NP.sum(NP.any(label>0, axis=-1))
        per = NP.sum(-1.0*label*NP.log(net_out+EPSI))/NP.sum(label>0)
        return acc, per
       

test_his = []
for i in xrange(200):
    for j in xrange(500):
        data, label, mask  = ptb.get_batch(train_batch_size)
        n_cost, net_out, net_dec = train_func(data, mask, label)
        acc, per = evaluate(net_out, label[:,1:])
        dec_acc, dec_per = evaluate(net_dec, label)
        print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' ACC = ', acc, ' Per = ', per, 'Dec ACC = ', dec_acc, ' Per = ', dec_per
 
    test_acc = []
    test_per = []
    test_dacc = []
    test_dper = []
    for j in xrange(ptb.test_size/test_batch_size):
        data, label, mask = ptb.get_batch(test_batch_size, test=True)
        n_cost, net_out, net_dec = test_func(data, mask, label)
        acc, per = evaluate(net_out, label[:,1:])
        dec_acc, dec_per = evaluate(net_dec, label)
        print ' Test Batch = ', j,  
        test_acc.append(acc)
        test_per.append(per)
        test_dacc.append(dec_acc)
        test_dper.append(dec_per)
    print '\nEpoch = ', str(i), ' Test Acc = ', NP.mean(NP.asarray(test_acc)), ' Test Per = ', NP.mean(NP.asarray(test_per)), 'Test Dec ACC = ', NP.mean(NP.asarray(test_dacc)), 'Test Dec Per = ', NP.mean(NP.asarray(test_dper))
    NP.savez(fname+'_test_show.npz', label=label, predict=net_out, dec=net_dec)
    test_his.append(NP.mean(NP.asarray(test_per)))
    model.save(fname+'_'+str(i))
model.save(fname)
NP.savez(fname+'_result.npz', test_his = test_his)

