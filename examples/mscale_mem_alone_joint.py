#the code may be quite ugly, but this version is using in now running experiments, the optimization will add in next version(such as speed up cross_entropy and remove abs in attention part)
import sys, getopt
import numpy as NP
import theano
import theano.tensor as T
import theano.tensor.nnet as NN
from wrapper import *
from txt_data_word import *

def to_one_hot(x, dim):
    y = NP.zeros((dim,), dtype=NP.int32)
    if x>=0:
        y[x]=1
    return y


try:
    opts, args = getopt.getopt(sys.argv[1:], "h", ["ptb", "bbc", "imdb", "wiki"])
except getopt.GetoptError:
    print 'Usage, please type --ptb, --bbc, --imdb, --wiki, to determine which dataset'
    sys.exit(2)

fname = './saves/mem_alone'
for opt, arg in opts:
    if opt == '-h':
        print 'Usage, please type --ptb, --bbc, --imdb, --wiki, to determine which dataset'
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

Dataset = txt_data(train_fname, test_fname, dict_fname, dataset_fnames, gen_dict=False, only_dict=True, DICT_SIZE = DICT_SIZE, raw_file = raw_file)
in_dim = Dataset.c_dim # character dict size
out_dim = Dataset.c_dim 
dict_size = 50 #our word emb dict is an 50 * 100 matrix
gru1_dim = 200 # gru in character LM 
gru2_dim = 200 # gru in word LM
word_out_dim = Dataset.w_dim # word dict size
emb_dim = 100 # word emb dim
train_batch_size = 32
test_batch_size = 128

char_in = T.tensor3() 
char_target = T.imatrix()
cw_mask = T.matrix() #character to word mask
cw_index1 = T.imatrix() #convert word embedding matrix from sparse to dense. using to speed up 
cw_index2 = T.imatrix()
word_target = T.imatrix()
model = Model()
#model.load(fname+'_'+str(229))

#NULL a b c I J    input
#a    b c I J K    output
#0    0 0 1 0 0 1  mask

#get mask for handle variable length in batch, all zeros vector will be masked out
def get_mask(cur_in, mask_value=0.):
    return T.shape_padright(T.any((1. - T.eq(cur_in, mask_value)), axis=-1))

#character step
def _step(char, mask, trash, prev_h1, word_emb, h_before, word_emb_base):
	batch_mask = get_mask(char)
	gru1 = batch_mask * model.gru(cur_in = char, rec_in = prev_h1, name = 'gru1', shape = (in_dim, gru1_dim))
	att = model.att_mem(cur_in = gru1-h_before, mem_in = word_emb_base, name = 'att1', shape = (gru1_dim, emb_dim), tick=dict_size) # the tick is useless in this task, i using it for variable length memory ...
        mask =T.shape_padright(mask) # using our word mask
        # mem means our word emb dict 
	mem_in = T.sum(word_emb_base.dimshuffle([1, 0, 2]) * att.dimshuffle([1, 0, 'x']), axis=1) # using the attention to extract word emb from dict
	word_emb = mem_in * mask + word_emb * (1-mask)
        h_before = gru1 * mask + h_before * (1-mask)
        fc1 = batch_mask * NN.softmax(model.fc(cur_in = gru1, name = 'fc_char', shape = (gru1_dim, out_dim)))

	return fc1, gru1, word_emb,h_before

def word_step(word_emb, trash, prev_h1):
	batch_mask = get_mask(word_emb)
	gru1 = batch_mask * model.gru(cur_in = word_emb, rec_in = prev_h1, name = 'gru_word', shape = (emb_dim, gru2_dim))
	fc1 = batch_mask * NN.softmax(model.fc(cur_in = gru1, name = 'fc_word', shape = (gru2_dim, word_out_dim)))
	return fc1, gru1
def help_crossentropy(matrix, ivector):
    mask = 1. - T.eq(ivector, -1)
    return mask * NN.categorical_crossentropy(matrix, ivector)
def idx_crossentropy(tensor, imatrix):
    sc, _ = theano.scan(help_crossentropy, sequences=[tensor, imatrix])
    return sc
#define the word emb dict
word_emb_base = model.Wmatrix(name='word_emb_base', shape=(dict_size, emb_dim)).dimshuffle(0, 'x', 1)
#scan for character step
sc, _ = theano.scan(_step, sequences=[char_in.dimshuffle([1, 0, 2]), cw_mask.dimshuffle([1, 0])], outputs_info = [T.zeros((char_in.shape[0], out_dim)),T.zeros((char_in.shape[0], gru1_dim)), T.zeros((char_in.shape[0], emb_dim)), T.zeros((char_in.shape[0], gru1_dim))], non_sequences=[word_emb_base])

char_out = sc[0].dimshuffle([1, 0, 2])
char_out = char_out[:, :-1]
EPSI = 1e-15
#loss for character LM, char_target is same with char_in, so it will be shuffle one step for LM task
cost_char_LM = T.sum(idx_crossentropy(T.clip(char_out, EPSI, 1.0-EPSI), char_target[:,1:]))/T.sum(char_target[:,1:]>=0)

#encoder information, word emb
word_embs = sc[-2].dimshuffle(1, 0, 2)
word_embs = word_embs[cw_index1, cw_index2]
word_embs = word_embs.dimshuffle(1, 0, 2)

#scan for word LM, char_in.shape[0] is the batch_size ...
sc,_ = theano.scan(word_step, sequences=[word_embs], outputs_info = [T.zeros((char_in.shape[0], word_out_dim)), T.zeros((char_in.shape[0], gru2_dim))])

word_out = (sc[0]).dimshuffle([1, 0, 2])

#loss for word LM task, word_target is tensor now, it will be matrix in next version
cost_word_LM = T.sum(idx_crossentropy(T.clip(word_out, EPSI, 1.0-EPSI), word_target))/T.sum(word_target>=0)

cost_all = cost_char_LM + cost_word_LM
test_func_all = theano.function([char_in, cw_mask, char_target, cw_index1, cw_index2, word_target], [cost_all, char_out, word_out], allow_input_downcast=True)
print 'TEST COMPILE'
grad_all = rmsprop(cost_all, model.weightsPack.getW_list(), lr=1e-3, epsilon=1e-6, ignore_input_disconnect=False)
train_func_all = theano.function([char_in, cw_mask, char_target, cw_index1, cw_index2, word_target], [cost_all, char_out, word_out], updates=grad_all, allow_input_downcast=True)
print 'TRAIN COMPILE'

#calculate the Acc and PPL, -1 means mask
def evaluate(net_out, label):
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==label).astype('int')+(label>=0).astype('int')>1)
        net_out += EPSI
        net_out /= NP.sum(net_out, axis=2)[:,:,NP.newaxis]
        label_one_hot = NP.asarray([[to_one_hot(x, NP.shape(net_out)[-1]) for x in batch] for batch in label])
        per = NP.sum(-1.0*label_one_hot*NP.log(net_out))
        return acc, per, NP.sum(label>=0)
       

for i in xrange(200):
    for j in xrange(1000):
        char, char_label, mask, word_index1, word_index2, word_label = Dataset.get_batch(train_batch_size)
        n_cost, char_out, word_out = train_func_all(char, mask, char_label, word_index1, word_index2, word_label)
        acc, per, cnt = evaluate(char_out, char_label[:,1:])
        wacc, wper, wcnt = evaluate(word_out, word_label)
        print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' Char Acc = ', 1.0*acc/cnt, ' BPC = ', (1.0*per/cnt)/NP.log(2), ' Word Acc = ', 1.0*wacc/wcnt, ' Word PPL = ', NP.exp(1.0*wper/wcnt), ' LEN = ', NP.shape(char)[1]
    test_acc = []
    test_bpc = []
    test_cnt = []
    test_wacc = []
    test_wper = []
    test_wcnt = []
    for j in xrange(Dataset.test_size/test_batch_size):
        char, char_label, mask, word_index1, word_index2, word_label = Dataset.get_batch(test_batch_size, test=True)
        n_cost, char_out, word_out = test_func_all(char, mask, char_label, word_index1, word_index2, word_label)
        acc, per, cnt = evaluate(char_out, char_label[:,1:])
        wacc, wper, wcnt = evaluate(word_out, word_label)
        test_acc.append(acc)
        test_bpc.append(per/NP.log(2))
        test_cnt.append(cnt)
        test_wacc.append(wacc)
        test_wper.append(wper)
        test_wcnt.append(wcnt)
        print ' Test Batch = ', str(j), 
    print '\nEpoch = ', str(i), ' Test char Acc = ', NP.sum(test_acc)/NP.sum(test_cnt), ' Test BPC = ', NP.sum(test_bpc)/NP.sum(test_cnt), ' Test word Acc = ', NP.sum(test_wacc)/NP.sum(test_wcnt), ' Test Word PPL = ', NP.exp(NP.sum(test_wper)/NP.sum(test_wcnt))
    model.save(fname+'_joint_'+str(i))

