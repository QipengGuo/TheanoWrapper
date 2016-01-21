import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import numpy as NP
from seg_data import *
from wrapper import *
from collections import OrderedDict
import sys
fname = 'seg_rnn_cpu'
train_batch_size = 1
test_batch_size = 1
test_his = []

#opt here

in_dim = 4
BiRNN_dim = 5
C_dim = 24
SEG_dim = 18
STK_dim = 5
DUR_MAX_VOCAB = 15
DUR_EMB_DIM = 4
TAG_MAX_VOCAB = 67
TAG_EMB_DIM = 32
max_time = 500
max_seg = 10
max_tag = 67
seg_data = Seg_Data()
#coords batch(must be 1) channels
word_seq = T.tensor3()
storke = T.ivector()
all_st = T.ivector()
all_ed = T.ivector()
all_st_notag = T.ivector()
all_ed_notag = T.ivector()
all_tag = T.ivector()
label_st = T.ivector()
label_ed = T.ivector()
label_tag = T.ivector()
model = Model()

def logsumexp(x):
    x_max = T.max(x)
    return T.log(T.sum(T.exp(x-x_max)))+x_max

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

def _step_BiRNN(cur_in, prev_h1):
    gru1 = model.gru(cur_in = cur_in, rec_in = prev_h1, name = 'gru_BiRNN', shape=(in_dim, BiRNN_dim))
    return gru1

def _step_to_C(cur_in, trash):
    fc1 = T.tanh(model.fc(cur_in = cur_in, name = 'fc_2C', shape=(BiRNN_dim*2, C_dim)))
    return fc1

def _step_SEG(st, ed, prev_h1, mem, inputs):
    gru1 = model.gru(cur_in = inputs[ed], rec_in = T.switch(T.eq((st-ed)**2,1), T.zeros_like(prev_h1), prev_h1), name='gru_SEG', shape=(C_dim, SEG_dim))
    mem = T.set_subtensor(mem[st*max_seg+ed], gru1)
    return gru1, mem

def _calc_prob(st, ed, tag, prob, temp_prob, last_ed, mem_SEG, mem_C):
    prob = T.switch(T.eq(ed, last_ed), prob, T.set_subtensor(prob[last_ed], T.max(temp_prob)))
    BiSEG = T.concatenate((mem_SEG[st*max_seg+ed], mem_SEG[ed*max_seg+st]), axis=1)
    context = T.concatenate((T.switch(st>0, mem_C[st-1], mem_C[0]), mem_C[ed-1]), axis=1)
    #context = T.concatenate((mem_C[st], mem_C[ed-1]), axis=1) # TODO add C_START and C_END
    duration_emb = model.embedding(cur_in  = T.clip(T.switch(st-ed>=0,st-ed, ed-st), 0, DUR_MAX_VOCAB), name = 'dur_emb', shape = (DUR_MAX_VOCAB, DUR_EMB_DIM))
    tag_emb = model.embedding(cur_in = tag, name = 'tag_emb', shape = (TAG_MAX_VOCAB, TAG_EMB_DIM))
    all_feat = T.concatenate((BiSEG, context, duration_emb, tag_emb), axis=1)
    p = NN.sigmoid(model.fc(cur_in = all_feat, name = 'fc_prob', shape = (SEG_dim*2+C_dim*2+DUR_EMB_DIM+TAG_EMB_DIM,1)))
    temp_idx = (ed-st-1)*max_tag+tag
    temp_idx = T.switch(temp_idx<max_tag*max_seg, temp_idx, max_tag*max_seg-1)
    temp_prob = T.set_subtensor(temp_prob[temp_idx], p+prob[st])
    return prob, temp_prob, ed

def _RNN_fwd(X):
    t = theano.shared(NP.zeros((1, BiRNN_dim)))
    sc, _ = theano.scan(_step_BiRNN, sequences=[X], outputs_info=[t])
    return sc
def _RNN_back(X):
    t = theano.shared(NP.zeros((1, BiRNN_dim)))
    sc, _ = theano.scan(_step_BiRNN, sequences=[X[::-1]], outputs_info=[t])
    return sc

def _storke_emb(pos, trash, X, storke):
    fwd = _RNN_fwd(X[storke[pos]:storke[pos+1]])
    back = _RNN_back(X[storke[pos]:storke[pos+1]])
    return T.concatenate((fwd[-1], back[-1]), axis=1)

t = T.unbroadcast(T.zeros((1, BiRNN_dim*2)), 0, 1)
storke_emb, _ = theano.scan(_storke_emb, sequences=[T.arange(storke.shape[0]-1)], outputs_info = [t], non_sequences=[word_seq, storke])
t = T.unbroadcast(T.zeros((1, C_dim)), 0, 1)
mem_C, _ = theano.scan(_step_to_C, sequences=[storke_emb], outputs_info = [t])

mem_SEG = theano.shared(NP.zeros((max_time * max_seg, 1, SEG_dim)))
def _SEG_emb(st, ed, mem_SEG, mem_C):
    t = T.unbroadcast(T.zeros((1, SEG_dim)), 0, 1)
    sc, _ = theano.scan(_step_SEG, sequences=[st, ed], outputs_info=[t, mem_SEG], non_sequences=[mem_C])
    mem_SEG = sc[1][-1]
    sd, _ = theano.scan(_step_SEG, sequences=[st[::-1], ed[::-1]], outputs_info=[t, mem_SEG], non_sequences=[mem_C])
    mem_SEG = sc[1][-1]
    return mem_SEG

mem_SEG = _SEG_emb(all_st_notag, all_ed_notag, mem_SEG, mem_C)

MAX_NUM = 9999
all_prob = theano.shared(NP.zeros((max_time, 1, 1)))
temp_prob = theano.shared(NP.zeros(((max_seg+1)*max_tag, 1, 1))-MAX_NUM)
label_prob = theano.shared(NP.zeros((max_time, 1, 1)))

def _calc(prob, temp_prob, st, ed, tag, mem_SEG, mem_C):
    sc, _ = theano.scan(_calc_prob, sequences=[st, ed, tag], outputs_info=[prob, temp_prob, ed[0]], non_sequences=[mem_SEG, mem_C])
    prob = sc[0][-1]
    return prob

all_prob = _calc(all_prob, temp_prob, all_st, all_ed, all_tag, mem_SEG, mem_C)
label_prob = _calc(label_prob, temp_prob, label_st, label_ed, label_tag, mem_SEG, mem_C)
all_Z = T.max(all_prob)
label_Z = T.max(label_prob)

cost = all_Z - label_Z


test_func = theano.function([word_seq, storke, all_st, all_ed, all_st_notag, all_ed_notag, all_tag, label_st, label_ed, label_tag], [cost, all_Z, label_Z], allow_input_downcast=True)
grad = rmsprop(cost, model.weightsPack.getW_list(), lr=1e-2, epsilon=1e-4)
train_func = theano.function([word_seq, storke, all_st, all_ed, all_st_notag, all_ed_notag, all_tag, label_st, label_ed, label_tag], [cost, all_Z, label_Z], updates=grad, allow_input_downcast=True)


for i in xrange(50):
    for j in xrange(50):
        X, stk, all_st, all_ed, all_st_notag, all_ed_notag, all_tag, label_st, label_ed, label_tag = seg_data.get_sample()
        n_cost, t1, t2 = train_func(X, stk, all_st, all_ed, all_st_notag, all_ed_notag, all_tag, label_st, label_ed, label_tag)
        print 'Epoch = ', i, ' Batch = ', j, ' Train Cost = ', n_cost, t1, t2
    test_cost = []
    for j in xrange(len(seg_data.test_X)):
        X, stk, all_st, all_ed, all_st_notag, all_ed_notag, all_tag, label_st, label_ed, label_tag = seg_data.get_sample(test=True)
        n_cost, t1, t2 = test_func(X, stk, all_st, all_ed, all_st_notag, all_ed_notag, all_tag, label_st, label_ed, label_tag)
        test_cost.append(n_cost)

    print 'Epoch = ', i, ' Test Cost = ', NP.mean(test_cost)
    model.save(fname+'_'+str(i))

model.save(fname)
NP.savez(fname+'_result.npz', test_his = test_his)



