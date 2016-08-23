import numpy as NP
import theano
import numpy.random as RNG

def ones(shape):
    return NP.cast[theano.config.floatX](NP.ones(shape))

def zeros(shape):
    return NP.cast[theano.config.floatX](NP.zeros(shape))

def uniform(shape, low=-0.5, high=0.5):
    return NP.cast[theano.config.floatX](RNG.uniform(low=low, high=high, size=shape))

def get_fans(shape):
    '''
    Borrowed from keras
    '''
    fan_in = shape[0] if len(shape) == 2 else NP.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]
    return fan_in, fan_out

def glorot_uniform(shape):
    '''
    Borrowed from keras
    '''
    fan_in, fan_out = get_fans(shape)
    s = NP.sqrt(6. / (fan_in + fan_out))
    return NP.cast[theano.config.floatX](RNG.uniform(low=-s, high=s, size=shape))

def orthogonal(shape, scale=1.1):
    '''
    Borrowed from keras
    '''
    flat_shape = (shape[0], NP.prod(shape[1:]))
    a = RNG.normal(0, 1, flat_shape)
    u, _, v = NP.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return NP.cast[theano.config.floatX](q)
