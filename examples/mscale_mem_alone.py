#the code may be quite ugly, but this version is using in now running experiments, the optimization will add in next version(such as speed up cross_entropy and remove abs in attention part)
import numpy as NP
import theano
import theano.tensor as T
import theano.tensor.nnet as NN
from wrapper import *
from ptb_data_word import *
fname = 'ptb_mem_alone'
ptb = ptb_data(write_dict=True)
in_dim = ptb.c_dim # character dict size
out_dim = ptb.c_dim 
DICT_SIZE = 50 #our word emb dict is an 50 * 100 matrix
gru1_dim = 200 # gru in character LM 
gru2_dim = 200 # gru in word LM
word_out_dim = ptb.w_dim # word dict size
mem_dim = 100 # word emb dim
train_batch_size = 32
test_batch_size = 32
mask_value = 1.0 # useless
char_in = T.tensor3() 
char_target = T.tensor3()
cw_mask = T.matrix() #character to word mask
word_target = T.tensor3()
model = Model()
#model.load(fname+'_'+str(229))

#NULL a b c I J    input
#a    b c I J K    output
#0    0 0 1 0 0 1  mask

#get mask for handle variable length in batch, all zeros vector will be masked out
def get_mask(cur_in, mask_value=0.):
    return T.shape_padright(T.any((1. - T.eq(cur_in, mask_value)), axis=-1))

#character step
def _step(char, mask, trash, prev_h1, word_emb, h_before, mem):
	batch_mask = get_mask(char)
	gru1 = batch_mask * model.gru(cur_in = char, rec_in = prev_h1, name = 'gru1', shape = (in_dim, gru1_dim))
	att = model.att_mem(cur_in = gru1-h_before, mem_in = mem, name = 'att1', shape = (gru1_dim, mem_dim), tick=DICT_SIZE) # the tick is useless in this task, i using it for variable length memory ...
        mask =T.shape_padright(mask) # using our word mask
        # mem means our word emb dict 
	mem_in = T.sum(mem.dimshuffle([1, 0, 2]) * att.dimshuffle([1, 0, 'x']), axis=1) # using the attention to extract word emb from dict
	word_emb = mem_in * mask + word_emb * (1-mask)
        h_before = gru1 * mask + h_before * (1-mask)
        fc1 = batch_mask * NN.softmax(model.fc(cur_in = gru1, name = 'fc1', shape = (gru1_dim, out_dim)))

	return fc1, gru1, word_emb,h_before

def word_step(word_emb, mask, trash, prev_h1):
	batch_mask = get_mask(mask)
        mask = T.shape_padright(mask)
	gru1 = batch_mask * model.gru(cur_in = word_emb, rec_in = prev_h1, name = 'gru_word', shape = (mem_dim, gru2_dim))
        # skip the time step which mask equal zero, this is the reason why we training so slowly
	next_h1 = mask * gru1 + (1-mask) * prev_h1
	fc1 = mask * batch_mask * NN.softmax(model.fc(cur_in = gru1, name = 'fc_word', shape = (gru2_dim, word_out_dim)))
	return fc1, next_h1

#define the word emb dict
mem = model.Wmatrix(name='mem', shape=(DICT_SIZE, mem_dim)).dimshuffle(0, 'x', 1)
#scan for character step
sc, _ = theano.scan(_step, sequences=[char_in.dimshuffle([1, 0, 2]), cw_mask.dimshuffle([1, 0])], outputs_info = [T.zeros((char_in.shape[0], out_dim)),T.zeros((char_in.shape[0], gru1_dim)), T.zeros((char_in.shape[0], mem_dim)), T.zeros((char_in.shape[0], gru1_dim))], non_sequences=[mem])

char_out = sc[0].dimshuffle([1, 0, 2])
char_out = char_out[:, :-1]
EPSI = 1e-15
#loss for character LM, char_target is same with char_in, so it will be shuffle one step for LM task
cost = T.mean(NN.categorical_crossentropy(T.clip(char_out, EPSI, 1.0-EPSI), char_target[:,1:]))

#encoder information, word emb
enc = sc[-2]

#scan for word LM, char_in.shape[0] is the batch_size ...
sc,_ = theano.scan(word_step, sequences=[enc, cw_mask.dimshuffle([1, 0])], outputs_info = [T.zeros((char_in.shape[0], word_out_dim)), T.zeros((char_in.shape[0], gru2_dim))])

word_out = (sc[0]).dimshuffle([1, 0, 2])

#loss for word LM task, word_target is tensor now, it will be matrix in next version
cost += T.mean(NN.categorical_crossentropy(T.clip(word_out, EPSI, 1.0-EPSI), word_target))


test_func = theano.function([char_in, cw_mask, char_target, word_target], [cost, char_out, word_out], allow_input_downcast=True)
print 'TEST COMPILE'
grad = rmsprop(cost, model.weightsPack.getW_list(), lr=1e-3, epsilon=1e-6)
train_func = theano.function([char_in, cw_mask, char_target, word_target], [cost, char_out, word_out], updates=grad, allow_input_downcast=True)
print 'TRAIN COMPILE'

#calculate the Acc and PPL
def evaluate(net_out, label):
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==NP.argmax(label, 2)).astype('int')+(NP.any(label>0, axis=-1)).astype('int')>1)
        net_out += EPSI
        net_out /= NP.sum(net_out, axis=2)[:,:,NP.newaxis]
        per = NP.sum(-1.0*label*NP.log(net_out))
        return acc, per, NP.sum(label>0)
       

test_his = []
for i in xrange(0, 500):
    for j in xrange(500):
        data, label, mask, word_label  = ptb.get_batch(train_batch_size)
        #print NP.shape(word_label)
        n_cost, net_out, net_dec = train_func(data, mask, label, word_label)
        acc, per, cnt = evaluate(net_out, label[:,1:]) # i should using BPC here later
        dec_acc, dec_per, dcnt = evaluate(net_dec, word_label) # wrong name from previous code, not decode, it is word Acc and word PPL
        print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' ACC = ', 1.0*acc/cnt, ' BPC = ', (1.0*per/cnt)/NP.log(2), 'Dec ACC = ', 1.0*dec_acc/dcnt, ' Per = ', NP.exp(1.0*dec_per/dcnt), ' LEN = ', NP.shape(data)[1]
 
    # test part, same with training
    test_acc = []
    test_per = []
    test_dacc = []
    test_dper = []
    test_cnt = []
    test_dcnt = []
    for j in xrange(ptb.test_size/test_batch_size):
        data, label, mask, word_label = ptb.get_batch(test_batch_size, test=True)
        n_cost, net_out, net_dec = test_func(data, mask, label, word_label)
        acc, per, cnt = evaluate(net_out, label[:,1:])
        dec_acc, dec_per, dcnt = evaluate(net_dec, word_label)
        print ' Test Batch = ', j,  
        test_acc.append(acc)
        test_per.append(per)
        test_cnt.append(cnt)
        test_dacc.append(dec_acc)
        test_dper.append(dec_per)
        test_dcnt.append(dcnt)
    print '\nEpoch = ', str(i), ' Test Acc = ', NP.sum(test_acc)/NP.sum(test_cnt), ' Test BPC = ', (NP.sum(test_per)/NP.sum(test_cnt))/NP.log(2), 'Test Dec ACC = ', NP.sum(test_dacc)/NP.sum(test_dcnt), 'Test Dec Per = ', NP.exp(NP.sum(test_dper)/NP.sum(test_dcnt))
    NP.savez(fname+'_test_show.npz', label=label, predict=net_out, dec=net_dec)
    test_his.append(NP.mean(NP.asarray(test_per)))
    model.save(fname+'_'+str(i))
model.save(fname)
NP.savez(fname+'_result.npz', test_his = test_his)

