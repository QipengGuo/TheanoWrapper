import numpy as NP
import theano
import theano.tensor as T
import theano.tensor.nnet as NN

def sigmoid(x):
	return NN.sigmoid(x)

def tanh(x):
	return T.tanh(x)

def softmax(x):
	return NN.softmax(x)

def softmax_fast(x):
	e_x = T.exp(x)
	sm = e_x / T.sum(e_x, axis=0, keepdims=True)
	return sm

def relu(x):
	return T.switch(x>0.0, x, 0.0)

def relu_leak(x, neg_coef=0.1):
	return T.swtich(x>0.0, x, neg_coef*x)
