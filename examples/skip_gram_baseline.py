import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import theano.tensor.extra_ops as EX
from wrapper import *
from ptb_skip_gram_data import *
fname='skip_gram_baseline_D'
ptb = ptb_skip_gram()
in_dim = ptb.wcnt
emb_dim = 100
train_batch_size = 32
test_batch_size = 32

word_idx_a = T.ivector()
word_idx_b = T.ivector()
prob_target = T.vector() # 1, -1

model = Model()
dict_in = model.Wmatrix(name='word_in', shape=(in_dim, emb_dim))
dict_out = model.Wmatrix(name='word_out', shape=(in_dim , emb_dim))

emb_in = T.dot(EX.to_one_hot(word_idx_a, in_dim), dict_in)
emb_out = T.dot(EX.to_one_hot(word_idx_b, in_dim), dict_out)

cost = -1.0* T.sum(T.log(NN.sigmoid(prob_target*T.sum(emb_in*emb_out, axis=1))))

train_func = theano.function([word_idx_a, word_idx_b, prob_target], cost, updates=rmsprop(cost, model.weightsPack.getW_list(), lr=1e-3, epsilon=1e-6), allow_input_downcast=True)

print 'Train compile'
test_func_in = theano.function([word_idx_a], emb_in, allow_input_downcast=True)
test_func_out = theano.function([word_idx_b], emb_out, allow_input_downcast=True)
print 'Test compile'
for i in xrange(50):
	for j in xrange(500):
		word_a, word_b, label = ptb.get_batch(train_batch_size)
		cost = train_func(word_a, word_b, label)
		print 'Epoch ', i, ' Batch ', j, ' Cost=', cost
	model.save(fname+'_'+str(i))
	
	
