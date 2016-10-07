# TheanoWrapper
An easy-to-use wrapper of theano

#How to use
```python

#these package are necessary in common
import theano 
import theano.tensor as T
import theano.tensor.nnet as NN
import numpy as NP

from wrapper import * #import tools

#prepare your data in numpy format 


#declare input and output variable
in_data = T.matrix() #a simple mlp network, two axis are (samples, channels)
label = T.matrix()

model = Model() #declare a model 


#define network working flow
def _step(X):
    fc1 = T.tanh(model.fc(X, name='fc1', shape=((in_dim, h_dim)))) #fc means fully connect layer
    fc2 = T.tanh(model.fc(fc1, name='fc2', shape=((h_dim, out_dim)))) 
    return fc2

net_out = _step(in_data) #get the output of network

loss = T.mean((net_out-label)**2) #define loss function

rms = rmsprop(cost, model.weightsPack.getW_list(), lr=1e-2, epsilon=1e-4) #define optimizer, getW_list() return all parameters, don't decalre unused layers in model 
train_func = theano.function([in_data, label], [loss, net_out], updates=rms, allow_input_downcast=True)
test_func = theano.function([in_data, label], [loss, net_out], allow_input_downcast=True)

model.load('abc') # load should appear after the model construction

#one epoch, seen each training sample once
train_cost, _ = auto_batch(train_func, train_batch_size, train_data, train_label) # train_data, train_label is the real data
test_cost, _ = auto_batch(test_func, test_batch_size, test_data, test_label)

model.save('abc') #save the model in h5py format
```

