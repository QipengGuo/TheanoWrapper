import time
import sys, getopt
import numpy as NP
import theano
import theano.tensor as T
import theano.tensor.nnet as NN
from wrapper import *
from collections import OrderedDict
from txt_data_word import *

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
    opts, args = getopt.getopt(sys.argv[1:], "h", ["ptb", "bbc", "imdb", "wiki", "de", "cs", "fr", "train", "test", "baseline", "context_free", 'context_dependent_ema', 'context_dependent_gru'])
except getopt.GetoptError:
    print 'Usage, please type --ptb, --bbc, --imdb, --wiki, to determine which dataset, --train or --test, and the model can be --baseline, --context_free, --context_dependent'
    sys.exit(2)


fname = './saves/mscale_drop_new'

gru1_dim = 200 # gru in character LM 
gru2_dim = 400 # gru in word LM
emb_dim = 200 # word emb dim, must equal to gru1_dim due to the high way network
char_emb_dim = 15
train_batch_size = 20
test_batch_size = 20
drop_flag = False

train_flag = False
mode = None
for opt, arg in opts:
    if opt == '-h':
        print 'Usage, please type --ptb, --bbc, --imdb, --wiki, to determine which dataset, --train or --test to select training (without validation) or testing (testing on validation set), the model type was --baseline, --context_free, --context_dependent'
        sys.exit(2)
    if opt == '--ptb':
        fname = fname + '_ptb'
        dict_fname = 'ptb_dict'
        train_fname = 'data/ptb_train.txt'
        test_fname = 'data/ptb_valid.txt'
        DICT_SIZE = 10000
        dataset_fnames = ['data/ptb_train.txt', 'data/ptb_valid.txt', 'data/ptb_test.txt']
        raw_file = False
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
        train_fname = 'data/imdb_train.txt'
        test_fname = 'data/imdb_valid.txt'
        DICT_SIZE = 30000
        dataset_fnames = ['data/imdb_train.txt', 'data/imdb_valid.txt', 'data/imdb_test.txt']
        raw_file = False
    if opt == '--cs':
        fname = fname + '_cs'
        dict_fname = 'cs_dict'
        train_fname = 'data/cs.train'
        test_fname = 'data/cs.valid'
        DICT_SIZE = 30000
        dataset_fnames = ['data/cs.train', 'data/cs.valid', 'data/cs.test']
        raw_file = False
    if opt == '--de':
        fname = fname + '_de'
        dict_fname = 'de_dict'
        train_fname = 'data/de.train'
        test_fname = 'data/de.valid'
        DICT_SIZE = 30000
        dataset_fnames = ['data/de.train', 'data/de.valid', 'data/de.test']
        raw_file = False
    if opt == '--fr':
        fname = fname + '_fr'
        dict_fname = 'fr_dict'
        train_fname = 'data/fr.train'
        test_fname = 'data/fr.valid'
        DICT_SIZE = 30000
        dataset_fnames = ['data/fr.train', 'data/fr.valid', 'data/fr.test']
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
    if opt == '--baseline':
        mode = 'baseline'
    if opt == '--context_free':
        mode = 'context_free'
    if opt == '--context_dependent_gru':
        mode = 'context_dependent_gru'
    if opt == '--context_dependent_ema':
        mode = 'context_dependent_ema'

fname = fname + '_' + mode
print 'File name = ', fname
Dataset = txt_data(train_fname, test_fname, dict_fname, dataset_fnames, gen_dict=False, only_dict=True, DICT_SIZE = DICT_SIZE, raw_file = raw_file)

#Dataset_test = txt_data(test_fname, 'data/ptb_test.txt', dict_fname, dataset_fnames, gen_dict=False, only_dict=True, DICT_SIZE = DICT_SIZE, raw_file = raw_file, batch_size=1)

in_dim = Dataset.c_dim # character dict size
out_dim = Dataset.c_dim 
word_out_dim = Dataset.w_dim # word dict size

var_lr = theano.shared(NP.asarray(1.0, dtype=theano.config.floatX)) #learning rate of rmsprop
var_rescale = theano.shared(NP.asarray(5.0, dtype=theano.config.floatX)) # rescale the gradient norm
char_in = T.imatrix()
cw_index1 = T.imatrix() #convert word embedding matrix from sparse to dense. using to speed up 
cw_index2 = T.imatrix()
word_in = T.imatrix()
word_target = T.imatrix()
model = Model()
#model.load(fname+'_150')
#model.load('./saves/mscale_de_context_free_34')
#drop1 = Dropout(shape=(train_batch_size, gru1_dim))
drop2 = Dropout(shape=(word_target.shape[0]*word_target.shape[1], emb_dim), prob=0.2)
drop3 = Dropout(shape=(word_target.shape[0], gru2_dim), prob=0.5)
drop4 = Dropout(shape=(word_target.shape[0], emb_dim), prob=0.5)
drop5 = Dropout(shape=(word_target.shape[0]*word_target.shape[1], gru2_dim), prob=0.5)

def categorical_crossentropy(prob, true_idx):
    true_idx = T.arange(true_idx.shape[0]) * word_out_dim + true_idx
    t1 = prob.flatten()[true_idx]
    return -t1

#NULL a b c I J    input
#a    b c I J K    output
#0    0 0 1 0 0 1  mask

#get mask for handle variable length in batch, all zeros vector will be masked out
def get_mask(x_in, mask_value=0.):
    return T.shape_padright(T.any((1. - T.eq(x_in, mask_value)), axis=-1))

#character step

def relu(x):
    return T.switch(x>0, x, 0.1*x)

def high_way(x_in=None, name='', shape=[]):
	g = sigmoid(model.fc(x_in = x_in, name=name+'_g', shape=shape))
	h = tanh(model.fc(x_in=x_in, name=name+'_h', shape=shape))
	return h*g + x_in * (1. - g)

def _word_step_embedding(word):
    batch_mask = get_mask(word, -1.)
    word_emb = batch_mask * model.embedding(x_in = word, name='word_embedding', shape=(word_out_dim, emb_dim))
    return word_emb

def _char_step_context_free(char, prev_h1):
	batch_mask = get_mask(char, -1.)
    char = model.embedding(x_in = char, name = 'char_emb', shape = (in_dim, char_emb_dim))
	gru1 = batch_mask * model.gru(x_in = char, rec_in = prev_h1, name = 'gru_char', shape = (char_emb_dim, gru1_dim))
	return gru1

def _gen_word_emb_step_ema(h_context_free, h_before):
    batch_mask = get_mask(h_context_free)
	D_h_c = h_before 
	D_h_cf = h_context_free
	D_h = concatenate([D_h_c, D_h_cf], axis=1)
	hw_h = high_way(x_in = D_h, name = 'hw_emb_h', shape=(emb_dim+gru1_dim, emb_dim+gru1_dim))
	word_emb = batch_mask * tanh(model.fc(x_in = hw_h, name = 'fc_emb', shape = (emb_dim+gru1_dim, emb_dim)))
	h_before = h_before * 0.5 + word_emb * 0.5
	return h_before, word_emb

def _gen_word_emb_step_gru(h_context_free, h_before):
    batch_mask = get_mask(h_context_free)
	D_h_c = h_before 
	D_h_cf = h_context_free
	D_h = concatenate([D_h_c, D_h_cf], axis=1)
	hw_h = high_way(x_in = D_h, name = 'hw_emb_h', shape=(emb_dim+gru1_dim, emb_dim+gru1_dim))
	word_emb = batch_mask * model.fc(x_in = hw_h, name = 'fc_emb', shape = (emb_dim+gru1_dim, emb_dim))
	gru_emb = model.gru(x_in = word_emb, rec_in = h_before, name = 'gru_emb', shape = (emb_dim, emb_dim))
	return gru_emb, word_emb

def _gen_word_emb_step_free(h_context_free):
    batch_mask = get_mask(h_context_free) 
	D_h_cf = h_context_free
	hw_cf = high_way(x_in = D_h_cf, name = 'hw_emb_cf', shape=(gru1_dim, emb_dim))
	word_emb = batch_mask * hw_cf
	return word_emb

def softmax(x):
    e_x = T.exp(x)
    sm = e_x / e_x.sum(axis=1, keepdims=True)
    return sm

def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def word_step(word_emb, prev_h1):
	batch_mask = get_mask(word_emb)

	gru1 = model.gru(x_in = word_emb, rec_in = prev_h1, name = 'gru_word', shape = (emb_dim, gru2_dim))

#        if drop_flag:
#            D_gru1 = drop5.drop(gru1)
#        else:
#            D_gru1 = gru1

#        gru2 = model.gru(x_in = D_gru1, rec_in = prev_h2, name = 'gru_word_2', shape = (gru2_dim, gru2_dim))

	return batch_mask * gru1 + (1-batch_mask) * prev_h1, gru1

drop_flag = False
EPSI = 1e-15

train_state_gen_emb = theano.shared(NP.zeros((train_batch_size , emb_dim), dtype=theano.config.floatX))
train_state_word_rnn = theano.shared(NP.zeros((train_batch_size , gru2_dim), dtype=theano.config.floatX))
train_state_word_rnn2 = theano.shared(NP.zeros((train_batch_size , gru2_dim), dtype=theano.config.floatX))

test_state_gen_emb = theano.shared(NP.zeros((test_batch_size , emb_dim), dtype=theano.config.floatX))
test_state_word_rnn = theano.shared(NP.zeros((test_batch_size , gru2_dim), dtype=theano.config.floatX))
test_state_word_rnn2 = theano.shared(NP.zeros((test_batch_size , gru2_dim), dtype=theano.config.floatX))


def get_express(train=False, emb_flag=None):
	global drop_flag
	drop_flag = train
	batch_size = word_target.shape[0]        
    state_updates = OrderedDict()
    state_gen_emb = train_state_gen_emb if train else test_state_gen_emb
    state_word_rnn = train_state_word_rnn if train else test_state_word_rnn
    word_embs=None
    if emb_flag in ['context_free', 'context_dependent_gru', 'context_dependent_ema']:
        sc, _ = theano.scan(_char_step_context_free, sequences=[char_in.dimshuffle(1,0)], outputs_info = [T.zeros((char_in.shape[0], gru1_dim))], name='scan_char_rnn', profile=False)
        # assign character time step to word time step
        h_context_free = sc.dimshuffle(1,0,2)
        h_context_free = h_context_free.reshape((sc.shape[0] * sc.shape[1], gru1_dim))
	    cw_index = cw_index1 * sc.shape[0] + cw_index2
        h_context_free = h_context_free[cw_index].reshape((cw_index.shape[0], cw_index.shape[1], gru1_dim))
        h_context_free = h_context_free.dimshuffle(1,0,2)
    
        if emb_flag=='context_dependent_ema':
            sc, _ = theano.scan(_gen_word_emb_step_ema, sequences=[h_context_free], outputs_info = [state_gen_emb, None], name='scan_gen_emb', profile=False)
            word_embs = sc[1]
            state_updates[state_gen_emb] = sc[0][-1]
        if emb_flag=='context_dependent_gru':
            sc, _ = theano.scan(_gen_word_emb_step_gru, sequences=[h_context_free], outputs_info = [state_gen_emb, None], name='scan_gen_emb', profile=False)
            word_embs = sc[1]
            state_updates[state_gen_emb] = sc[0][-1]

        if emb_flag=='context_free':
            sc, _ = theano.scan(_gen_word_emb_step_free, sequences=[h_context_free], name='scan_gen_emb_free')
            word_embs = sc
#                word_embs = h_context_free
        if emb_flag=='baseline':
            sc, _ = theano.scan(_word_step_embedding, sequences=[word_in.dimshuffle(1,0)], name='scan_word_emb')
            word_embs = sc

        if drop_flag:
            d_word_embs = drop2.drop(word_embs.reshape((-1, emb_dim)))
            word_embs = d_word_embs.reshape((word_embs.shape[0], word_embs.shape[1], word_embs.shape[2]))
            sc,_ = theano.scan(word_step, sequences=[word_embs], outputs_info = [state_word_rnn, None], name='scan_word_rnn', profile=False, truncate_gradient=-1)
        state_updates[state_word_rnn] = sc[0][-1]

	word_out = sc[-1].dimshuffle(1,0,2).reshape((word_target.shape[0]*(word_target.shape[1]), gru2_dim))
    if drop_flag:
        word_out = drop5.drop(word_out)
	word_out = log_softmax(model.fc(x_in = word_out, name = 'fc_word', shape=(gru2_dim, word_out_dim)))
	#word_out = T.clip(word_out, EPSI, 1.0-EPSI)

	f_word_target = word_target.reshape((word_target.shape[0]*(word_target.shape[1]), ))
	PPL_word_LM = T.sum((1. -T.eq(f_word_target, -1)) * categorical_crossentropy(word_out, f_word_target))
	cost_word_LM = PPL_word_LM/T.sum(f_word_target>=0)
	cost_all = cost_word_LM

	if train:
            #grad_all, grad_norm = rmsprop(cost_all, model.weightsPack.getW_list(), lr=var_lr,epsilon=var_lr**2, rescale = var_rescale , ignore_input_disconnect=True)
            #state_updates = OrderedDict()
            grad_all, grad_norm = pure_sgd(cost_all, model.weightsPack.getW_list(), lr=var_lr,rescale = var_rescale, ignore_input_disconnect=False)
		return cost_all, PPL_word_LM, merge_OD(grad_all, state_updates), grad_norm
	else:
		return cost_all, PPL_word_LM, state_updates
	
cost_all, PPL, all_updates = get_express(train=False, emb_flag=mode)
if mode=='baseline':
    test_func_hw = theano.function([word_in, word_target], [cost_all, PPL], updates=all_updates, allow_input_downcast=True)
if mode in ['context_free', 'context_dependent_ema', 'context_dependent_gru']:
    test_func_hw = theano.function([char_in, cw_index1, cw_index2, word_target], [cost_all, PPL], updates=all_updates, allow_input_downcast=True)

print 'TEST COMPILE'

if train_flag :
    cost_all, PPL, all_updates, grad_norm = get_express(train=True, emb_flag=mode)
    if mode=='baseline':
        train_func_hw = theano.function([word_in, word_target], [cost_all, PPL, grad_norm], updates=all_updates, allow_input_downcast=True)
    if mode in ['context_free', 'context_dependent_ema', 'context_dependent_gru']:
        train_func_hw = theano.function([char_in, cw_index1, cw_index2, word_target], [cost_all, PPL], updates=all_updates, allow_input_downcast=True)
    print 'TRAIN COMPILE'       

train_func = None
test_func = None

def reset_states():
    #train_state_char_rnn.set_value(train_state_char_rnn.get_value()*0.0)
    #train_state_gen_emb.set_value(train_state_gen_emb.get_value()*0.0)
    #train_state_word_rnn.set_value(train_state_word_rnn.get_value()*0.0)
    test_state_gen_emb.set_value(test_state_gen_emb.get_value()*0.0)
    test_state_word_rnn.set_value(test_state_word_rnn.get_value()*0.0)

last_valid_PPL = 2e10
valid_increase_cnt = 0 #if loss on validation set didn't decrease in N steps, change the learning rate as half
reset_states()
for i in xrange(0, 50):
    train_func = train_func_hw if train_flag else None
    test_func = test_func_hw
    if train_flag:
        ma_cost = Moving_AVG(1327) 
        mytime = time.time()
        train_batchs = int(Dataset.train_size/train_batch_size)
        for j in xrange(train_batchs):
            if mode=='baseline':
                word, word_label = Dataset.get_batch(only_word=True)
                n_cost, n_ppl, n_norm = train_func(word, word_label)
                ma_cost.append(n_cost)
                print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' PPL = ', NP.exp(n_ppl/NP.sum(word_label>=0)), ' AVG Cost = ', ma_cost.get_avg(), 'LEN = ', NP.shape(word_label)[1], 'GRAD Norm = ', n_norm

            if mode in ['context_free', 'context_dependent_ema', 'context_dependent_gru']:
                if mode in [ 'context_dependent_ema', 'context_dependent_gru']:
                    var_rescale.set_value(2.0) #3.0 ema 2.0 gru
                else:
                    var_rescale.set_value(5.0)
                char, word_index1, word_index2, word_label = Dataset.get_batch()
                n_cost, n_ppl = train_func(char, word_index1, word_index2, word_label)
                ma_cost.append(n_cost)
                print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' PPL = ', NP.exp(n_ppl/NP.sum(word_label[:,1:]>=0)), ' AVG Cost = ', ma_cost.get_avg(), 'LEN = ', NP.shape(char)[1], NP.shape(word_index1)[1]

    #        print train_func.profile.summary()
        newtime = time.time()
        print 'One Epoch Time = ', newtime-mytime
        mytime = newtime
        model.save(fname+'_'+str(i))
    if not train_flag:
        model.load(fname+'_'+str(i))
    reset_states()
    Dataset.test_data_idx = 0
    Dataset.test_len_idx = 0
    test_wper = []
    test_wcnt = []
    test_batchs = 0
    while test_batchs<Dataset.test_size:
        if mode=='baseline':
            word, word_label = Dataset.get_batch(test=True, only_word=True)
            n_cost, n_ppl = test_func(word, word_label)
        if mode in ['context_free', 'context_dependent_ema', 'context_dependent_gru']:
            char, word_index1, word_index2, word_label = Dataset.get_batch(test=True)
            n_cost, n_ppl = test_func(char, word_index1, word_index2, word_label)
        test_wper.append(n_ppl)
        test_wcnt.append(NP.sum(word_label>=0))
        test_batchs+=len(word_label)
        print ' Test Progress = ', 1.0*test_batchs/Dataset.test_size, 
    valid_PPL = NP.exp(NP.sum(test_wper)/NP.sum(test_wcnt))
    print '\nEpoch = ', str(i), ' Test Word PPL = ', valid_PPL
    if valid_PPL+1 > last_valid_PPL:
        valid_increase_cnt += 1
        if valid_increase_cnt>=1:
            print 'change learning rate', var_lr.get_value()*0.5
            var_lr.set_value(var_lr.get_value()*0.5)
            valid_increase_cnt = 0
    last_valid_PPL = valid_PPL



