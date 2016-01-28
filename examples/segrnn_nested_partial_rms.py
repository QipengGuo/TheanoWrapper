import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import theano.gradient as GRAD
import numpy as NP
from seg_data_nested import *
from wrapper import *
from collections import OrderedDict
import sys
fname = 'seg_rnn_nest_rms_partial'
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
MAX_NUM = 9999
seg_data = Seg_Data()
#coords batch(must be 1) channels
word_seq = T.tensor3()
storke = T.ivector()
tag_list = T.ivector()
match_ref = T.tensor3()
label_seq = T.ivector()
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

def _RNN_fwd(X):
    t = theano.shared(NP.zeros((1, BiRNN_dim), dtype=theano.config.floatX))
    sc, _ = theano.scan(_step_BiRNN, sequences=[X], outputs_info=[t])
    return sc
def _RNN_back(X):
    t = theano.shared(NP.zeros((1, BiRNN_dim), dtype=theano.config.floatX))
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
mem_C = T.concatenate((mem_C, T.zeros((1, 1, C_dim))), axis=0)

def _step_SEG(idx, prev_h1, mem, inputs, start):
    gru1 = model.gru(cur_in = inputs[idx], rec_in = prev_h1, name='gru_SEG', shape=(C_dim, SEG_dim))
    mem = T.set_subtensor(mem[start, idx-start], gru1)
    return gru1, mem

def loop1_seg(idx, mem, inputs, lens):
    t = T.switch(idx+max_seg<lens, idx+max_seg, lens)
    sc, _ = theano.scan(_step_SEG, sequences=[T.arange(idx, t)], outputs_info=[T.unbroadcast(T.zeros((1, SEG_dim)), 0, 1), mem], non_sequences=[inputs, idx])
    return sc[1][-1]

def loop1_seg_rev(idx, mem, inputs, lens):
    t = T.switch(idx+max_seg<lens, idx+max_seg, lens)
    sc, _ = theano.scan(_step_SEG, sequences=[T.arange(t-1, idx-1, step=-1)], outputs_info=[T.unbroadcast(T.zeros((1, SEG_dim)), 0, 1), mem], non_sequences=[inputs, idx])
    return sc[1][-1]

def loop2_seg(loop_func, lens, mem, inputs):
    sc, _ = theano.scan(loop_func, sequences=[T.arange(lens)], outputs_info=[mem], non_sequences=[inputs, lens])
    return sc[-1]

#mem_SEG = theano.shared(NP.zeros((max_time, max_seg, 1, SEG_dim), dtype=theano.config.floatX))
mem_SEG = T.unbroadcast(T.zeros((storke.shape[0], max_seg, 1, SEG_dim)), 0, 1, 2, 3)
def _SEG_emb(lens, mem_SEG, mem_C):
    mem_SEG_l = loop2_seg(loop1_seg, lens, mem_SEG, mem_C)
    mem_SEG_r = loop2_seg(loop1_seg_rev, lens, mem_SEG, mem_C)
    return T.concatenate((mem_SEG_l, mem_SEG_r), axis=3)
mem_SEG = _SEG_emb(storke.shape[0], mem_SEG, mem_C)

#t_func = theano.function([word_seq, storke], mem_SEG, allow_input_downcast=True)
#X, stk, _, _ = seg_data.get_sample()
#print NP.shape(t_func(X, stk))
def _calc_prob(st, mem_SEG, mem_C, ed, tag, prob, label_prob, match_ref):
    BiSEG = mem_SEG[st, ed-st-1]
    context = T.concatenate((T.switch(st>0, mem_C[st-1], mem_C[0]), T.switch(ed<mem_C.shape[0]-1, mem_C[ed], mem_C[ed-1])), axis=1) # TODO add C_START and C_END
    duration_emb = model.embedding(cur_in  = T.clip(T.switch(st-ed>=0,st-ed, ed-st), 0, DUR_MAX_VOCAB), name = 'dur_emb', shape = (DUR_MAX_VOCAB, DUR_EMB_DIM))
    tag_emb = model.embedding(cur_in = tag, name = 'tag_emb', shape = (TAG_MAX_VOCAB, TAG_EMB_DIM))
    all_feat = T.concatenate((BiSEG, context, duration_emb, tag_emb), axis=1)
    p = T.tanh(model.fc(cur_in = all_feat, name = 'fc_prob', shape = (SEG_dim*2+C_dim*2+DUR_EMB_DIM+TAG_EMB_DIM,32))) #relu may cause all zeros
    p2 = T.tanh(model.fc(cur_in = p, name = 'fc2_prob', shape = (32, 32)))
    p3 = model.fc(cur_in =p2, name = 'fc3_prob', shape = (32, 1))
    return p3+prob[st], T.switch(T.eq(match_ref[st, ed, tag], 1), p3+label_prob[st], -MAX_NUM)

def _calc_prob_decode(st, mem_SEG, mem_C, ed, tag, prob):
    BiSEG = mem_SEG[st, ed-st-1]
    context = T.concatenate((T.switch(st>0, mem_C[st-1], mem_C[0]), T.switch(ed<mem_C.shape[0]-1, mem_C[ed], mem_C[ed-1])), axis=1) # TODO add C_START and C_END
    duration_emb = model.embedding(cur_in  = T.clip(T.switch(st-ed>=0,st-ed, ed-st), 0, DUR_MAX_VOCAB), name = 'dur_emb', shape = (DUR_MAX_VOCAB, DUR_EMB_DIM))
    tag_emb = model.embedding(cur_in = tag, name = 'tag_emb', shape = (TAG_MAX_VOCAB, TAG_EMB_DIM))
    all_feat = T.concatenate((BiSEG, context, duration_emb, tag_emb), axis=1)
    p = T.tanh(model.fc(cur_in = all_feat, name = 'fc_prob', shape = (SEG_dim*2+C_dim*2+DUR_EMB_DIM+TAG_EMB_DIM,32))) #relu may cause all zeros
    p2 = T.tanh(model.fc(cur_in = p, name = 'fc2_prob', shape = (32, 32)))
    p3 = model.fc(cur_in =p2, name = 'fc3_prob', shape = (32, 1))
    return p3+prob[st]

def _calc_label_prob_one_label(idx, ed, st, tag, p, label_seq, label_prob):
    cond1 = T.switch(T.eq(label_seq[idx-1], tag)+T.eq(idx, 1)>=2, p, -MAX_NUM)
    cond2 = T.switch(T.eq(label_seq[idx-1], tag)+(idx>1)+(idx-1<=st)+((st-1)/max_seg < idx-1)>=4, p+label_prob[st, idx-1], -MAX_NUM)
    cond3 = T.switch(T.eq(st, 0), cond1, cond2)
    return T.switch(idx<=ed, cond3, -MAX_NUM)

def _calc_label_prob_partial(p, ed, st, tag, label_prob, label_seq):
    sc, _ = theano.scan(_calc_label_prob_one_label, sequences=[T.arange(1, label_seq.shape[0]+1)], non_sequences=[ed, st, tag, p, label_seq, label_prob])
    return sc

def _calc_prob_partial(st, mem_SEG, mem_C, ed, tag, prob, label_prob, label_seq):
    BiSEG = mem_SEG[st, ed-st-1]
    context = T.concatenate((T.switch(st>0, mem_C[st-1], mem_C[0]), T.switch(ed<mem_C.shape[0]-1, mem_C[ed], mem_C[ed-1])), axis=1) # TODO add C_START and C_END
    duration_emb = model.embedding(cur_in  = T.clip(T.switch(st-ed>=0,st-ed, ed-st), 0, DUR_MAX_VOCAB), name = 'dur_emb', shape = (DUR_MAX_VOCAB, DUR_EMB_DIM))
    tag_emb = model.embedding(cur_in = tag, name = 'tag_emb', shape = (TAG_MAX_VOCAB, TAG_EMB_DIM))
    all_feat = T.concatenate((BiSEG, context, duration_emb, tag_emb), axis=1)
    p = T.tanh(model.fc(cur_in = all_feat, name = 'fc_prob', shape = (SEG_dim*2+C_dim*2+DUR_EMB_DIM+TAG_EMB_DIM,32))) #relu may cause all zeros
    p2 = T.tanh(model.fc(cur_in = p, name = 'fc2_prob', shape = (32, 32)))
    p3 = model.fc(cur_in =p2, name = 'fc3_prob', shape = (32, 1))
    return p3+prob[st], _calc_label_prob_partial(p3, ed, st, tag, label_prob, label_seq)

def loop1_prob(tag, mem_SEG, mem_C, ed, prob, label_prob, match_ref):
    t = T.switch(ed-max_seg>=0, ed-max_seg, 0)
    sc, _ = theano.scan(_calc_prob, sequences=[GRAD.disconnected_grad(T.arange(t, ed))], non_sequences=[mem_SEG, mem_C, ed, tag, prob, label_prob, match_ref])
    return T.max(sc[0]), T.max(sc[1])

def loop1_decode(tag, mem_SEG, mem_C, ed, prob):
    t = T.switch(ed-max_seg>=0, ed-max_seg, 0)
    sc, _ = theano.scan(_calc_prob_decode, sequences=[T.arange(t, ed)], non_sequences=[mem_SEG, mem_C, ed, tag, prob])
    return T.max(sc), 1.0*(T.argmax(sc)+t)
    #return sc[0]

def loop1_partial(tag, mem_SEG, mem_C, ed, prob, label_prob, label_seq):
    t = T.switch(ed-max_seg>=0, ed-max_seg, 0)
    sc, _ = theano.scan(_calc_prob_partial, sequences=[GRAD.disconnected_grad(T.arange(t, ed))], non_sequences=[mem_SEG, mem_C, ed, tag, prob, label_prob, label_seq])
    return T.max(sc[0]), sc[1]

def loop2_prob(ed, prob, label_prob, mem_SEG, mem_C, match_ref, tag_list):
    sc, _ = theano.scan(loop1_prob, sequences=[tag_list], non_sequences=[mem_SEG, mem_C, ed, prob, label_prob, match_ref])
    prob = T.set_subtensor(prob[ed], T.max(sc[0]))
    label_prob = T.set_subtensor(label_prob[ed], T.max(sc[1]))
    return prob, label_prob

def loop2_decode(ed, prob, mem_SEG, mem_C, tag_list):
    sc, _ = theano.scan(loop1_decode, sequences=[tag_list], non_sequences=[mem_SEG, mem_C, ed, prob])
    return 1.0*T.argmax(sc[0]), sc[1][T.argmax(sc[0])], T.set_subtensor(prob[ed], T.max(sc[0]))

def loop2_partial(ed, prob, label_prob, mem_SEG, mem_C, label_seq, tag_list):
    sc, _ = theano.scan(loop1_partial, sequences=[tag_list], non_sequences=[mem_SEG, mem_C, ed, prob, label_prob, label_seq])
    prob = T.set_subtensor(prob[ed], T.max(sc[0]))
    label_prob = T.set_subtensor(label_prob[ed], (T.max(sc[1], axis=(0,1))).reshape((-1, 1)))
    return prob, label_prob

def loop3_prob(lens, mem_SEG, mem_C, match_ref, tag_list):
    sc, _ = theano.scan(loop2_prob, sequences=[T.arange(1, lens)], outputs_info=[T.unbroadcast(T.zeros((lens, 1)), 0, 1), T.unbroadcast(T.zeros((lens, 1)), 0, 1)], non_sequences=[mem_SEG, mem_C, match_ref, tag_list])
    return sc[0][-1], sc[1][-1]

def loop3_decode(lens, mem_SEG, mem_C, tag_list):
    all_prob = T.unbroadcast(T.zeros((lens, 1)), 0, 1)
    sc, _ = theano.scan(loop2_decode, sequences=[T.arange(1, lens)], outputs_info=[None, None, all_prob], non_sequences=[mem_SEG, mem_C, tag_list])
    return sc[0], sc[1]

def loop3_partial(lens, mem_SEG, mem_C, label_seq, tag_list):
    sc, _ = theano.scan(loop2_partial, sequences=[T.arange(1, lens)], outputs_info=[T.unbroadcast(T.zeros((lens, 1)), 0, 1), T.unbroadcast(T.zeros((lens, label_seq.shape[0], 1)), 0, 1, 2)], non_sequences=[mem_SEG, mem_C, label_seq, tag_list])
    return sc[0][-1], sc[1][-1]

#all_prob, label_prob = loop3_prob(storke.shape[0], mem_SEG, mem_C, match_ref, tag_list)

#all_Z = all_prob[-1]
#label_Z = label_prob[-1]

all_prob, label_prob = loop3_partial(storke.shape[0], mem_SEG, mem_C, label_seq, tag_list)

all_Z = all_prob[-1]
label_Z = label_prob[-1][-1]

cost = T.sum(all_Z - label_Z)

test_func = theano.function([word_seq, storke, tag_list, label_seq], [cost, all_Z, label_Z], allow_input_downcast=True)
print 'TEST DONE'
grad = rmsprop(cost, model.weightsPack.getW_list())
#train_func = theano.function([word_seq, storke, tag_list, match_ref], [cost, all_Z, label_Z], updates=grad, allow_input_downcast=True)
train_partial_func = theano.function([word_seq, storke, tag_list, label_seq], [cost, all_Z, label_Z], updates=grad, allow_input_downcast=True)
print 'TRAIN DONE'
dec_tag, dec_st = loop3_decode(storke.shape[0], mem_SEG, mem_C, tag_list)
decode_func = theano.function([word_seq, storke, tag_list], [dec_tag, dec_st], allow_input_downcast=True)
print 'DECODE DONE'


for i in xrange(100):
    for j in xrange(500):
        X, stk, all_tag, label_seq = seg_data.get_sample(partial=True)
        n_cost, t1, t2 = train_partial_func(X, stk, all_tag, label_seq)
        dec_tag, dec_st = decode_func(X, stk, all_tag)
        print 'Epoch = ', i, ' Batch = ', j, ' Train Cost = ', n_cost, t1, t2
        print dec_tag
        print dec_st
    test_cost = []
    for j in xrange(len(seg_data.test_X)):
        X, stk, all_tag, label_seq = seg_data.get_sample(partial=True, test=True)
        n_cost, t1, t2 = test_func(X, stk, all_tag, label_seq)
        test_cost.append(n_cost)
        dec_tag, dec_st = decode_func(X, stk, all_tag)
        print 'Test', j
        print dec_tag
        print dec_st

    print 'Epoch = ', i, ' Test Cost = ', NP.mean(test_cost)
    model.save(fname+'_'+str(i))

model.save(fname)
NP.savez(fname+'_result.npz', test_his = test_his)



