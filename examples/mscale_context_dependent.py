#the code may be quite ugly, but this version is using in now running experiments, the optimization will add in next version(such as speed up cross_entropy and remove abs in attention part)
import time
import sys, getopt
import numpy as NP
import theano
#theano.config.allow_gc=False
import theano.tensor as T
import theano.tensor.nnet as NN
from wrapper import *
from txt_data_word import *

def to_one_hot(x, dim):
    y = NP.zeros((dim,), dtype=NP.int32)
    if x>=0:
        y[x]=1
    return y

def regular_loss(W_list, batch_size):
    cost = 0.
    for i in W_list:
        cost = cost + REGULAR_FACTOR * 0.5 * T.sum(i**2) / batch_size
    return cost

def softmax_sig(x):
    s_x = NN.sigmoid(x)
    sm = s_x / T.sum(s_x, axis=0, keepdims=True)
    return sm

class Moving_AVG(object):
	def __init__(self, array_size=500):
		self.array_size=array_size
		self.queue = NP.zeros((array_size, ))
		self.idx = 0
		self.filled = False
	def append(self, x):
		self.queue[self.idx]=x
		self.idx = (self.idx + 1)%self.array_size
		if not self.filled and self.idx==0:
			self.filled = True
	def get_avg(self):
		result = NP.mean(self.queue) if self.filled else NP.sum(self.queue)/NP.sum(self.queue!=0)
		return result

try:
    opts, args = getopt.getopt(sys.argv[1:], "h", ["ptb", "bbc", "imdb", "wiki", "train", "test"])
except getopt.GetoptError:
    print 'Usage, please type --ptb, --bbc, --imdb, --wiki, to determine which dataset'
    sys.exit(2)


fname = './saves/context_dependent_bdrop_cemb_hw'

#REGULAR_FACTOR = 0.0005 # L2 norm loss
dict_size = 50 #our word emb dict is an 50 * 100 matrix
gru1_dim = 200 # gru in character LM 
gru2_dim = 400 # gru in word LM
emb_dim = 200 # word emb dim, must equal to gru1_dim due to the high way network
char_emb_dim = 15
train_batch_size = 128
test_batch_size = 128
drop_flag = False

train_flag = False
for opt, arg in opts:
    if opt == '-h':
        print 'Usage, please type --ptb, --bbc, --imdb, --wiki, to determine which dataset, --train or --test to select training (without validation) or testing (testing on validation set)'
        sys.exit(2)
    if opt == '--ptb':
        fname = fname + '_ptb'
        dict_fname = 'ptb_dict'
        train_fname = 'ptb_train.txt'
        test_fname = 'ptb_valid.txt'
        DICT_SIZE = 10000
        dataset_fnames = ['ptb_train.txt', 'ptb_valid.txt', 'ptb_test.txt']
        raw_file = True
    if opt == '--bbc':
        fname = fname + '_bbc'
        dict_fname = 'bbc_dict'
        train_fname = 'BBC_train.txt'
        test_fname = 'BBC_valid.txt'
        DICT_SIZE = 10000
        dataset_fnames = ['BBC_train.txt', 'BBC_valid.txt', 'BBC_test.txt']
        raw_file = False
    if opt == '--imdb':
        fname = fname + '_imdb'
        dict_fname = 'imdb_dict'
        train_fname = 'imdb_train.txt'
        test_fname = 'imdb_valid.txt'
        DICT_SIZE = 30000
        dataset_fnames = ['imdb_train.txt', 'imdb_valid.txt', 'imdb_test.txt']
        raw_file = False
    if opt == '--wiki':
        fname = fname + '_wiki'
        dict_fname = 'wiki_dict'
        train_fname = 'wiki_train.txt'
        test_fname = 'wiki_valid.txt'
        DICT_SIZE = 30000
        dataset_fnames = ['wiki_train.txt', 'wiki_valid.txt', 'wiki_test.txt']
        raw_file = True
    if opt == '--train':
        train_flag = True
    if opt == '--test':
        train_flag = False

Dataset = txt_data(train_fname, test_fname, dict_fname, dataset_fnames, gen_dict=False, only_dict=True, DICT_SIZE = DICT_SIZE, raw_file = raw_file)
in_dim = Dataset.c_dim # character dict size
out_dim = Dataset.c_dim 
word_out_dim = Dataset.w_dim # word dict size

char_in = T.imatrix()
cw_mask = T.matrix() #character to word mask
cw_index1 = T.imatrix() #convert word embedding matrix from sparse to dense. using to speed up 
cw_index2 = T.imatrix()
word_target = T.imatrix()
model = Model()
#drop1 = Dropout(shape=(train_batch_size, gru1_dim))
drop2 = Dropout(shape=(train_batch_size, emb_dim))
drop3 = Dropout(shape=(train_batch_size, gru2_dim))
drop4 = Dropout(shape=(train_batch_size, emb_dim))

def categorical_crossentropy(prob, true_idx):
    true_idx = T.arange(true_idx.shape[0]) * word_out_dim + true_idx
    t1 = prob.flatten()[true_idx]
    return -T.log(t1)

#NULL a b c I J    input
#a    b c I J K    output
#0    0 0 1 0 0 1  mask

#get mask for handle variable length in batch, all zeros vector will be masked out
def get_mask(cur_in, mask_value=0.):
    return T.shape_padright(T.any((1. - T.eq(cur_in, mask_value)), axis=-1))

#character step

def high_way(cur_in=None, name='', shape=[]):
	g = NN.sigmoid(model.fc(cur_in = cur_in, name=name+'_g', shape=shape))
	h = T.tanh(model.fc(cur_in=cur_in, name=name+'_h', shape=shape))
	return h*g + cur_in * (1. - g)

def _char_step_context_free(char, mask, prev_h1):
	batch_mask = get_mask(char, -1.)
	mask = T.shape_padright(mask)
        char = model.embedding(cur_in = char, name = 'char_emb', shape = (in_dim, char_emb_dim))
	gru1 = batch_mask * model.gru(cur_in = char, rec_in = prev_h1, name = 'gru_char', shape = (char_emb_dim, gru1_dim))
	return (1. - mask) * gru1, gru1

def _gen_word_emb_step(h_context_free, h_before):
        batch_mask = get_mask(h_context_free)
	D_h_c = h_before 
	D_h_cf = h_context_free
	hw_c = high_way(cur_in = D_h_c, name = 'hw_emb_c', shape=(emb_dim ,emb_dim))
        hw_c = high_way(cur_in = hw_c, name = 'hw_emb_c2', shape=(emb_dim, emb_dim))
	hw_cf = high_way(cur_in = D_h_cf, name = 'hw_emb_cf', shape=(gru1_dim, emb_dim))
        hw_cf = high_way(cur_in = hw_cf, name = 'hw_emb_cf2', shape=(emb_dim, emb_dim))
	word_emb_att = batch_mask * 0.5 * (hw_c+hw_cf)
        #if drop_flag:
        if False:
            D_emb = drop4.drop(word_emb_att)
        else:
            D_emb = word_emb_att
	#gru_emb = model.gru(cur_in = D_emb, rec_in = h_before, name = 'gru_emb', shape = (emb_dim, emb_dim))
	#return gru_emb, word_emb_att, att
	h_before = h_before * 0.5 + D_emb * 0.5
	return h_before, word_emb_att

def softmax(x):
    e_x = T.exp(x)
    sm = e_x / e_x.sum(axis=1, keepdims=True)
    return sm

#import theano.sandbox.cuda.dnn as CUDNN
#dnnsoftmax =  CUDNN.GpuDnnSoftmax('bc01', 'fast', 'channel')
#def softmax(x):
#    ret =dnnsoftmax(x.dimshuffle(0, 1, 'x', 'x'))
#    return ret[:, :, 0, 0]

def word_step(word_emb, prev_h1):
	batch_mask = get_mask(word_emb)
        if drop_flag:
            D_word_emb = drop2.drop(word_emb)
        else:
            D_word_emb = word_emb

	gru1 = model.gru(cur_in = D_word_emb, rec_in = prev_h1, name = 'gru_word', shape = (emb_dim, gru2_dim))

	if drop_flag:
            D_gru1 = drop3.drop(gru1)
	else:
            D_gru1 = gru1

	return D_gru1

drop_flag = False
EPSI = 1e-15
def get_express(train=False, emb_flag=None):
	global drop_flag
	drop_flag = train
	batch_size = char_in.shape[0]
	sc, _ = theano.scan(_char_step_context_free, sequences=[char_in.dimshuffle(1,0), cw_mask.dimshuffle(1,0)], outputs_info = [T.zeros((batch_size, gru1_dim)), None], name='scan_char_rnn', profile=False)

        # assign character time step to word time step
        h_context_free = sc[1].dimshuffle(1,0,2)
	h_context_free = h_context_free.reshape((sc[1].shape[0] * sc[1].shape[1], gru1_dim))
	cw_index = cw_index1 * sc[1].shape[0]+ cw_index2
	h_context_free = h_context_free[cw_index].reshape((cw_index.shape[0], cw_index.shape[1], gru1_dim))
	h_context_free = h_context_free.dimshuffle(1,0,2)
        
        sc, _ = theano.scan(_gen_word_emb_step, sequences=[h_context_free], outputs_info = [T.zeros((batch_size, emb_dim)), None], name='scan_gen_emb', profile=False)

	word_embs = sc[1]

	sc,_ = theano.scan(word_step, sequences=[word_embs], outputs_info = [T.zeros((batch_size, gru2_dim))], name='scan_word_rnn', profile=False)

	word_out = sc.dimshuffle(1,0,2).reshape((cw_index1.shape[0]*cw_index1.shape[1], gru2_dim))
	word_out = softmax(model.fc(cur_in = word_out, name = 'fc_word', shape=(gru2_dim, word_out_dim)))
	word_out = T.clip(word_out, EPSI, 1.0-EPSI)

	f_word_target = word_target[:,1:].reshape((cw_index1.shape[0]*cw_index1.shape[1], ))
	PPL_word_LM = T.sum((1. -T.eq(f_word_target, -1)) * categorical_crossentropy(word_out, f_word_target))
	cost_word_LM = PPL_word_LM/T.sum(f_word_target>=0)
	cost_all = cost_word_LM

	if train:
                grad_all = rmsprop(cost_all, model.weightsPack.getW_list(), lr=1e-3,epsilon=1e-6, ignore_input_disconnect=True)
		return cost_all, PPL_word_LM, grad_all
	else:
		return cost_all, PPL_word_LM
	
cost_all, PPL = get_express(train=False)
test_func_hw = theano.function([char_in, cw_mask, cw_index1, cw_index2, word_target], [cost_all, PPL], allow_input_downcast=True)

print 'TEST COMPILE'

if train_flag :
    cost_all, PPL, grad_all = get_express(train=True)
    train_func_hw = theano.function([char_in, cw_mask, cw_index1, cw_index2, word_target], [cost_all, PPL], updates=grad_all, allow_input_downcast=True)
    print 'TRAIN COMPILE'       

train_func = None
test_func = None
for i in xrange(600):
    train_func = train_func_hw
    test_func = test_func_hw

    ma_cost = Moving_AVG(500)
    mytime = time.time()
    if train_flag:
        train_batchs = Dataset.train_size/train_batch_size
        train_batchs = min(2000, train_batchs)  
        for j in xrange(train_batchs):
            data_time = time.time()
            char, char_label, mask, word_index1, word_index2, word_label = Dataset.get_batch(train_batch_size)
            print 'One Data Time = ', time.time()-data_time
            batch_time = time.time()
            n_cost, n_ppl = train_func(char, mask, word_index1, word_index2, word_label)
            print 'One Batch Time = ', time.time() - batch_time
            ma_cost.append(n_cost)
            print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' PPL = ', NP.exp(n_ppl/NP.sum(word_label[:,1:]>=0)), ' AVG Cost = ', ma_cost.get_avg(), 'LEN = ', NP.shape(char)[1]

        #print train_func.profile.summary()
	newtime = time.time()
	print 'One Epoch Time = ', newtime-mytime
	mytime = newtime
        model.save(fname+'_'+str(i))
    if not train_flag:
        model.load(fname+'_'+str(i))
    Dataset.test_data_idx = 0
    Dataset.test_len_idx = 0
    test_wper = []
    test_wcnt = []
    test_batchs = Dataset.test_size/test_batch_size
    test_batchs = min(test_batchs, 50) if train_flag else test_batchs
    for j in xrange(test_batchs):
        char, char_label, mask, word_index1, word_index2, word_label = Dataset.get_batch(test_batch_size, test=True)
        n_cost, n_ppl = test_func(char, mask, word_index1, word_index2, word_label)

        test_wper.append(n_ppl)
        test_wcnt.append(NP.sum(word_label[:,1:]>=0))
        print ' Test Batch = ', str(j), 
    print '\nEpoch = ', str(i), ' Test Word PPL = ', NP.exp(NP.sum(test_wper)/NP.sum(test_wcnt))


