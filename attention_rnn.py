import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import numpy as NP
#TODO import data
from Wrapper import Model, rmsprop

fname = 'att_rnn'
train_batch_size = 16
test_batch_size = 32
test_his = []

#opt here

amazon_data = Amazon_data('item.json.gz', 'wordsEn.txt', cache=True)

word_seq = T.tensor3()
label_seq = T.tensor3()
in_dim = amazon_data.vocab_size
out_dim = amazon_data.vocab_size
starts = T.matrix()
model = Model()

def batched_dot(A, B):
		#borrowed from Draw, Google Deep mind group
		C = A.dimshuffle([0, 1, 2, 'x']) * B.dimshuffle([0, 'x', 1, ,2])
		return C.sum(axis=-2)
		
def get_mask(cur_in, mask_value=0.):
    return T.shape_padright(T.any((1. - T.eq(cur_in, mask_value)), axis=-1))
def masked(cur_in, mask):
    return cur_in * mask
def _step(cur_in, trash, prev_h1):
    mask = get_mask(cur_in)
    att = 'TODO'
    att = T.unbroadcast(att.dimshuffle([0, 1, 'x']), axis=2)
    rec_in1 = batched_dot(att,mem_bank)
    gru1 = masked(model.gru(cur_in = cur_in, rec_in = rec_in1, name = 'gru1', shape = (in_dim, 256)), mask)
    #gru2 = masked(model.gru(cur_in = gru1, rec_in = prev_h2, name = 'gru2', shape = (200, 200)), mask)
    #gru3 = masked(model.gru(cur_in = gru2, rec_in = prev_h3, name = 'gru3', shape = (200, 200)), mask)

    fc1 = masked(NN.softmax(model.fc(cur_in = gru1, name = 'fc1', shape = (256, out_dim))), mask)

		new_mem_bank = T.set_subtensor(mem_bank[time_step], gru1)
    return fc1, gru1
_word_seq = word_seq.dimshuffle(1, 0, 2)
#sc, _ = theano.scan(_step, sequences=[_word_seq], outputs_info=[starts, T.zeros((word_seq.shape[0], 200)), T.zeros((word_seq.shape[0], 200)), T.zeros((word_seq.shape[0], 200))], truncate_gradient=200)
sc, _ = theano.scan(_step, sequences=[_word_seq], outputs_info=[starts, T.zeros((word_seq.shape[0], 256))], truncate_gradient=-1)
word_out = sc[0].dimshuffle(1, 0, 2)

EPSI = 1e-6
cost = T.sum(NN.categorical_crossentropy(T.clip(word_out, EPSI, 1.0-EPSI), label_seq))
test_func = theano.function([word_seq, label_seq, starts], [cost, word_out], allow_input_downcast=True)
train_func = theano.function([word_seq, label_seq, starts], [cost, word_out], updates=rmsprop(cost, model.weightsPack.getW_list(), lr=1e-2, epsilon=1e-4), allow_input_downcast=True)
