import numpy as NP
import numpy.random as RNG
import theano
import theano.tensor as T
import theano.tensor.nnet as NN
from wrapper import * # import our tool

WORD_DIM = 1000 # vocab size
EMB_DIM = 100 # embedding size
H_DIM = 200 # gru units

seq_in = T.imatrix() # input sequence
seq_tar = T.imatrix() # target sequence

# define a model
model = Model()
seq_embs = model.embedding(seq_in, 'emb', (WORD_DIM, EMB_DIM))
gru_h = model.gru_flat(seq_embs, 'gru', (EMB_DIM, H_DIM))
seq_pred = softmax(model.fc(gru_h, 'softmax_fc', (H_DIM, WORD_DIM)).reshape((-1, WORD_DIM)))
NLL_loss = T.mean(NN.categorical_crossentropy(seq_pred, seq_tar.reshape((-1,))))

# backward
grads = rmsprop(NLL_loss, model.get_params())

#compile function
train_func = theano.function([seq_in, seq_tar], NLL_loss, updates=grads+model.get_updates(), allow_input_downcast=True)
test_func = theano.function([seq_in, seq_tar], NLL_loss, updates=model.get_updates(), allow_input_downcast=True)

#prepare data, replace it by your own real data
batch_size = 32
seq_len = 35
train_X = RNG.randint(0, WORD_DIM, (batch_size, seq_len+1))
train_Y = train_X[:,1:]
train_X = train_X[:,:-1]

# running 
n_cost = train_func(train_X, train_Y)
print n_cost

# save and load
#model.save('abc')
#model.load('abc')
