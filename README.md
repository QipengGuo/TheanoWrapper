# TheanoWrapper
Try to make some useful wrappers of theano for RNN 

# Need theano git version, not 0.7.0 release

# Some Examples
```python
#this code show a example for word-level language model on amazon reviews dataset
import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import numpy as NP
from apr_data import * #data parser
from Wrapper import Model, rmsprop  #import our tools

#setting
fname = 'apr_W' #path to store model and history
train_batch_size = 16
test_batch_size = 32
test_his = []

#init data parser
amazon_data = Amazon_data('', '', cache=True) #using exist cache

#model design
word_seq = T.tensor3()
label_seq = T.tensor3()
in_dim = amazon_data.vocab_size
out_dim = amazon_data.vocab_size
starts = T.matrix() #be all zero in this experiments
model = Model()

#model.load(fname) load the stored weights

#You need a mask for variance input length for batch   
def get_mask(cur_in, mask_value=0.):
	return T.shape_padright(T.any((1. - T.eq(cur_in, mask_value)), axis=-1))

def _step(cur_in, trash, prev_h1, prev_h2, prev_h3):  #cur_in -- current input, trash -- the prev final output uesless in this experiments, prev_h1, prev_h2, prev_h3 -- prev hidden state for GRU
	mask = get_mask(cur_in)
	gru1 = mask * model.gru(cur_in = cur_in, rec_in = prev_h1, name = 'gru1', shape=(in_dim, 200))
	gru2 = mask * model.gru(cur_in = gru1, rec_in = prev_h2, name = 'gru2', shape=(200, 200))
	gru3 = mask * model.gru(cur_in = gru2, rec_in = prev_h3, name = 'gru3', shape=(200,200))
	#these design a three layer gru network, the same name layer will share the weights
	
	fc1 = mask * NN.softmax(model.fc(cur_in = gru3, name = 'fc1', shape =(200, out_dim)))
	
	return fc1, gru1, gru2, gru3 #be the next trash, prev_h1, prev_h2, prev_h3
	
_word_seq = word_seq.dimshuffle(1, 0, 2) #shuffle the time axis to the first
sc, _ = theano.scan(_step, sequences=[_word_seq], outputs_info=[starts, T.zeros((word_seq.shape[0], 200)), T.zeros((word_seq.shape[0], 200)), T.zeros((word_seq.shape[0], 200))], truncate_gradient=-1) #truncate_gradient be k(k>0), means bp through k time steps
word_out = sc[0].dimshuffle(1, 0, 2) #recover the axis to be (batch_size, time, channel)

EPSI = 1e-6
cost = T.sum(NN.categorical_crossentropy(T.clip(word_out, EPSI, 1.0-EPSI), label_seq))
test_func = theano.function([word_seq, label_seq, starts], [cost, word_out], allow_input_downcast=True)
train_func = theano.function([word_seq, label_seq, starts], [cost, word_out], updates=rmsprop(cost, model.weightsPack.getW_list(), lr=1e-2, epsilon=1e-4), allow_input_downcast=True)
#the whole network has compiled into two function, test and train


#training process
for i in xrange(50):
    for j in xrange(200):
        data, label, n_len = amazon_data.get_batch(train_batch_size)
        n_cost, net_out = train_func(data, label, NP.zeros((train_batch_size, len(data[0][0]))))
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==NP.argmax(label, 2)).astype('int')+(NP.any(label>0, axis=-1)).astype('int')>1)/NP.sum(NP.any(label>0, axis=-1))
        per = 1.0 * NP.exp(NP.mean(NP.sum(-1.0*label*NP.log(net_out+EPSI), axis=(1, 2))/n_len))
        print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' ACC = ', acc, ' Per = ', per, ' Sent Len = ', NP.mean(n_len)
 
    test_acc = []
    test_per = []
    for j in xrange(amazon_data.test_size/test_batch_size):
        data, label, n_len = amazon_data.get_batch(test_batch_size, test=True)
        n_cost, net_out = test_func(data, label, NP.zeros((test_batch_size, len(data[0][0]))))
        acc = 1.0 * NP.sum((NP.argmax(net_out, 2)==NP.argmax(label, 2)).astype('int')+(NP.any(label>0, axis=-1)).astype('int')>1)/NP.sum(NP.any(label>0, axis=-1))
        per = 1.0 * NP>exp(NP.mean(NP.sum(-1.0*label*NP.log(net_out+EPSI), axis=(1, 2))/n_len))
        print ' Test Batch = ', j,  
        test_acc.append(acc)
        test_per.append(per)
    print '\nEpoch = ', str(i), ' Test Acc = ', NP.mean(NP.asarray(test_acc)), ' Test Per = ', NP.mean(NP.asarray(test_per))
    NP.savez('apr_test_show.npz', label=label, predict=net_out)
    test_his.append(NP.mean(NP.asarray(test_per)))
    model.save(fname)
model.save(fname)
NP.savez(fname+'_result.npz', test_his = test_his)
```