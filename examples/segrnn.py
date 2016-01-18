import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import numpy as NP
from seg_data import *
from wrapper import *
from collections import OrderedDict
import sys
fname = 'seg_rnn'
train_batch_size = 16
test_batch_size = 16
test_his = []

#opt here

in_dim = 4
BiRNN_dim =5
C_dim = 10
SEG_dim = 6
DUR_MAX_VOCAB = 15
DUR_EMB_DIM = 5
TAG_MAX_VOCAB = 67
TAG_EMB_DIM = 10
max_time = 100
seg_data = Seg_Data()

word_seq = T.tensor3()
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
    return T.log(T.sum(T.exp(x)))

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
    mem = T.set_subtensor(mem[st*max_time+ed], gru1)
    return gru1, mem

def _calc_prob(st, ed, tag, prob, last_ed, mem_SEG, mem_C):
    prob = T.switch(T.eq(ed, last_ed), prob, T.set_subtensor(prob[last_ed], T.log(prob[last_ed]+1e-6)))
    BiSEG = T.concatenate((mem_SEG[st*max_time+ed], mem_SEG[ed*max_time+st]), axis=1)
    context = T.concatenate((T.switch(st>0, mem_C[st-1], mem_C[0]), mem_C[ed-1]), axis=1)
    #context = T.concatenate((mem_C[st], mem_C[ed-1]), axis=1) # TODO add C_START and C_END
    duration_emb = model.embedding(cur_in  = T.clip(T.switch(st-ed>=0,st-ed, ed-st), 0, DUR_MAX_VOCAB), name = 'dur_emb', shape = (DUR_MAX_VOCAB, DUR_EMB_DIM))
    tag_emb = model.embedding(cur_in = tag, name = 'tag_emb', shape = (TAG_MAX_VOCAB, TAG_EMB_DIM))
    all_feat = T.concatenate((BiSEG, context, duration_emb, tag_emb), axis=1)
    p = NN.sigmoid(model.fc(cur_in = all_feat, name = 'fc_prob', shape = (SEG_dim*2+C_dim*2+DUR_EMB_DIM+TAG_EMB_DIM,1)))
    prob = T.set_subtensor(prob[ed], T.exp(p+prob[st]))
    return prob, ed

sc, _ = theano.scan(_step_BiRNN, sequences=[word_seq], outputs_info=[T.zeros((word_seq.shape[1], int(BiRNN_dim)))])
BiRNN_r = sc
sc, _ = theano.scan(_step_BiRNN, sequences=[word_seq[::-1]], outputs_info=[T.zeros((word_seq.shape[1], int(BiRNN_dim)))])
BiRNN_l = sc
sc, _ = theano.scan(_step_to_C, sequences=[T.concatenate((BiRNN_l, BiRNN_r), axis=2)], outputs_info = [T.zeros((word_seq.shape[1], int(C_dim)))])
mem_C = sc

mem_SEG = theano.shared(NP.zeros((max_time * max_time, 1, SEG_dim)))
sc, _ = theano.scan(_step_SEG, sequences=[all_st_notag, all_ed_notag], outputs_info = [T.zeros((word_seq.shape[1], SEG_dim)), mem_SEG], non_sequences=[mem_C])
mem_SEG = sc[1][-1]
sc, _ = theano.scan(_step_SEG, sequences=[all_st_notag[::-1], all_ed_notag[::-1]], outputs_info = [T.zeros((word_seq.shape[1], int(SEG_dim))), mem_SEG], non_sequences=[mem_C])
mem_SEG = sc[1][-1]

all_prob = theano.shared(NP.ones((max_time, 1, 1)))
sc, _ = theano.scan(_calc_prob, sequences=[all_st, all_ed, all_tag], outputs_info = [all_prob, all_ed[0]], non_sequences=[mem_SEG, mem_C])
all_prob = sc[0][-1]
label_prob = theano.shared(NP.ones((max_time, 1, 1)))
sc, _ = theano.scan(_calc_prob, sequences=[label_st, label_ed, label_tag], outputs_info = [label_prob, label_ed[0]], non_sequences=[mem_SEG, mem_C])
label_prob = sc[0][-1]
all_Z = T.max(all_prob)
label_Z = T.max(label_prob)

cost = all_Z - label_Z


test_func = theano.function([word_seq, all_st, all_ed, all_st_notag, all_ed_notag, all_tag, label_st, label_ed, label_tag], [cost, all_Z, label_Z], allow_input_downcast=True)
grad = rmsprop(cost, model.weightsPack.getW_list(), lr=1e-2, epsilon=1e-4)
train_func = theano.function([word_seq, all_st, all_ed, all_st_notag, all_ed_notag, all_tag, label_st, label_ed, label_tag], [cost, all_Z, label_Z], updates=grad, allow_input_downcast=True)


for i in xrange(1):
    for j in xrange(1):
        X, all_st, all_ed, all_st_notag, all_ed_notag, all_tag, label_st, label_ed, label_tag = seg_data.get_sample()
        n_cost, t1, t2 = train_func(X, all_st, all_ed, all_st_notag, all_ed_notag, all_tag, label_st, label_ed, label_tag)
        print 'Epoch = ', i, ' Batch = ', j, ' Train Cost = ', n_cost, t1, t2

model.save(fname)
NP.savez(fname+'_result.npz', test_his = test_his)



