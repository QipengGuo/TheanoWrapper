import numpy as NP
import numpy.random as RNG
import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import initialization as INIT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal.pool import pool_2d
#if theano.config.device[:3]=='cpu':
#    from theano.tensor.shared_randomstreams import RandomStreams
#if theano.config.device[:3]=='gpu':
#    from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams

class Dropout(object):
	def __init__(self, shape = None, prob=0.5):
		self.retain_prob = 1.0 - prob
		self.shape = shape
                self.seed = RNG.randint(1e6)
                self.rng = RandomStreams(self.seed)

	def drop(self, cur_in):
                self.mask = T.switch(self.rng.uniform(self.shape, dtype=theano.config.floatX)<self.retain_prob, 1., 0.)
	        h = cur_in * self.mask
		h /= self.retain_prob
		return h


def get_init(init_name):
    init_method = getattr(INIT, init_name, None)
    assert init_method is not None
    return init_method

class Ops_with_weights(object):
    def __init__(self, op_name = None):
        self.op_name = op_name
        self.created = False

        self.p = self.perform

    def create(self, save_weights=True, init_list=[], name_list=[], shape_list=[]):
        if self.created:
            return 

        if len(init_list)==1 and len(name_list)>1:
            init_list = [init_list[0]]*len(name_list)
        self.name_list = name_list
        self.shape_list = shape_list
        self.init_list = init_list
        self.params = [None]*len(init_list)
        self.updates = []
        assert len(init_list)==len(name_list) and len(name_list)==len(shape_list),  'init_list length must equals name_list and shape_list'
        for i in xrange(len(init_list)):
            self.params[i] = theano.shared(get_init(init_list[i])(shape_list[i]), name=name_list[i])
        self.save_weights=save_weights
        self.created = True
        print 'Create Ops ', name_list

    def reinit(self):
        for i in xrange(len(self.init_list)):
            self.params[i].set_value(get_init(self.init_list[i])(self.shape_list[i]))

    def perform(self):
        return NotImplemented

    def get_updates(self):
        return self.updates

    def get_params(self):
        return self.params

    def load(self, data):
        finds = 0
        for i in xrange(len(self.params)):
            target_name = self.name_list[i]
            target_shape = self.shape_list[i]
            for p in data:
                if p.name == target_name and p.get_value().shape == target_shape:
                    self.params[i].set_value(p.get_value())
                    finds += 1
                    break
        return finds == len(self.params)

class LinearLN(Ops_with_weights):
    def __init__(self, name=None, save_weights=True, shape=[], with_bias=True, init_list=None):
	init_list = ['glorot_uniform', 'zeros'] if init_list is None else init_list
        assert len(init_list)==2
        assert len(shape)==2
        super(self.__class__, self).__init__(name)

        in_dim, out_dim = shape
        self.with_bias = with_bias
        if with_bias:
            shape_list = [(in_dim, out_dim), (out_dim,)]
            name_list = [name+'_W', name+'_b']
        else:
            shape_list = [(in_dim, out_dim)]
            name_list = [name+'_W']
            init_list = init_list[0:1]
        shape_list = shape_list + [(in_dim,), (in_dim,)]
        init_list = init_list + ['ones', 'zeros']
        name_list = name_list + [name+'_S', name+'_Lb']
        self.create(save_weights=save_weights, init_list=init_list, name_list=name_list, shape_list=shape_list)

    def perform(self, x):
        EPSI = 1e-5
        W = self.params[0]
        if self.with_bias:
            b = self.params[1]
            S = self.params[2]
            Lb = self.params[3]
            x_ln = (x - T.mean(x, axis=-1, keepdims=True))/T.sqrt(T.var(x, axis=-1, keepdims=True)+EPSI)
            if x.ndim==3:
                x_ln = x_ln * S.dimshuffle('x', 'x', 0) + Lb.dimshuffle('x', 'x', 0)
                return batched_dot3(x_ln, W.dimshuffle('x', 0, 1)) + b.dimshuffle('x', 0)
            else:
                x_ln = x_ln * S.dimshuffle('x', 0) + Lb.dimshuffle('x', 0)
                return T.dot(x_ln, W) + b
        else:
            S = self.params[1]
            Lb = self.params[2]
            x_ln = (x - T.mean(x, axis=-1, keepdims=True))/T.sqrt(T.var(x, axis=-1, keepdims=True)+EPSI)
            if x.ndim==3:
                x_ln = x_ln * S.dimshuffle('x', 'x', 0) + Lb.dimshuffle('x', 'x', 0)
                return batched_dot3(x_ln, W.dimshuffle('x', 0, 1))
            else:
                x_ln = x_ln * S.dimshuffle('x', 0) + Lb.dimshuffle('x', 0)
                return T.dot(x_ln, W)

class Linear(Ops_with_weights):
    def __init__(self, name=None, save_weights=True, shape=[], with_bias=True, init_list=None):
	init_list = ['glorot_uniform', 'zeros'] if init_list is None else init_list
        assert len(init_list)==2
        assert len(shape)==2
        super(self.__class__, self).__init__(name)

        in_dim, out_dim = shape
        self.with_bias = with_bias
        if with_bias:
            shape_list = [(in_dim, out_dim), (out_dim,)]
            name_list = [name+'_W', name+'_b']
        else:
            shape_list = [(in_dim, out_dim)]
            name_list = [name+'_W']
            init_list = init_list[0:1]
        self.create(save_weights=save_weights, init_list=init_list, name_list=name_list, shape_list=shape_list)

    def perform(self, x):
        W = self.params[0]
        if self.with_bias:
            b = self.params[1]
            if x.ndim==3:
                return batched_dot3(x, W.dimshuffle('x', 0, 1)) + b.dimshuffle('x', 0)
            else:
                return T.dot(x, W) + b
        else:
            if x.ndim==3:
                return batched_dot3(x, W.dimshuffle('x', 0, 1))
            else:
                return T.dot(x, W)


class LayerNorm(Ops_with_weights):
    def __init__(self, name=None, save_weights=True, shape=[], init_list=None):
	init_list = ['ones', 'zeros'] if init_list is None else init_list
        assert len(init_list)==2
        assert len(shape)==1
        super(self.__class__, self).__init__(name)

        dim = shape
        shape_list = [(dim,), (dim,)]
        name_list = [name+'_S', name+'_b']

        self.create(save_weights=save_weights, init_list=init_list, name_list=name_list, shape_list=shape_list)

    def perform(self, x):
        EPSI = 1e-5

        S = self.params[0]
        b = self.params[1]

        x_ln = (x - T.mean(x_ln, axis=-1, keepdims=True))/T.sqrt(T.var(x, axis=-1, keepdims=True)+EPSI)
        if x.ndim==3:
            return x_ln * S.dimshuffle('x', 'x', 0) + b.dimshuffle('x', 'x', 0)
        else:
            return x_ln * S.dimshuffle('x', 0) + b.dimshuffle('x', 0)


conv2d = NN.conv2d
if theano.config.device[:3] == 'gpu':
    import theano.sandbox.cuda.dnn as CUDNN
    if CUDNN.dnn_available():
        print 'Using CUDNN instead of Theano conv2d'
        conv2d = CUDNN.dnn_conv

class Conv(Ops_with_weights):
    def __init__(self, name=None, save_weights=True, shape=[], init_list=None):
	init_list = ['glorot_uniform'] if init_list is None else init_list
        assert len(init_list)==1
        assert len(shape)==6
        super(self.__class__, self).__init__(name)
        
        nr_filters, nr_channels, filter_row, filter_col, self.conv_stride_row, self.conv_stride_col = shape
        shape_list = [(nr_filters, nr_channels, filter_row, filter_col)]
        name_list = [name+'_conv_W']
        self.create(save_weights=save_weights, init_list=init_list, name_list=name_list, shape_list=shape_list)

    def perform(self, x):
        conv_h = conv2d(x, self.params[0], subsample=(self.conv_stride_row, self.conv_stride_col), border_mode='half')
        return conv_h

class Pooling(object):
    def __init__(self, name=None, shape=[]):
        assert len(shape)==2
        self.name=name
        self.pool_shape=shape
        self.p=self.perform

    def perform(self, x):
        pool_h = pool_2d(x, self.pool_shape, ignore_border=True)
        return pool_h

    def get_updates(self):
        return []

    def get_params(self):
        return []

    def load(self, data):
        return 
   

class Fetch(Ops_with_weights):
    def __init__(self, name=None, save_weights=True, shape=[], init_list=None):
	init_list = ['uniform'] if init_list is None else init_list
        assert len(init_list)==1
        #assert len(shape)==2
        super(self.__class__, self).__init__(name)

        shape_list = [shape]
        name_list = [name+'D']
        self.create(save_weights=save_weights, init_list=init_list, name_list=name_list, shape_list=shape_list)

    def perform(self, x=None, get_all=False):
        assert x or get_all
        D = self.params[0]
        if get_all:
            return D
        else:
            return D[x]

class H_Softmax(Ops_with_weights):
    def __init__(self, name=None, save_weights=True, shape=[], init_list=None):
	init_list = ['glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform'] if init_list is None else init_list
        assert len(init_list)==4
        assert len(shape)==2
        super(self.__class__, self).__init__(name)

        in_dim, out_dim = shape
        best_match = [1, out_dim]
        for i in xrange(1, int(out_dim/2+2)):
            if i*int(out_dim/i)==out_dim and abs(i-int(out_dim/i))<abs(best_match[0]-best_match[1]):
                best_match = [i, int(out_dim/i)]

        shape_list = [(in_dim, best_match[0]), (best_match[0],), (best_match[0], in_dim, best_match[1]), (best_match[0], best_match[1])]
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_level1_size=best_match[0]
        self.h_level2_size=best_match[1]
        print 'H_softmax, two level ', best_match[0], best_match[1]
        name_list = [name+'_hf_level1_W', name+'_hf_level1_b', name+'_hf_level2_W', name+'_hf_level2_b']
        self.create(save_weights=save_weights, init_list=init_list, name_list=name_list, shape_list=shape_list)

    def perform(self, x, y=None):
        if y is not None:
            return NN.h_softmax(x, x.shape[0], self.out_dim, self.h_level1_size, self.h_level2_size, self.params[0], self.params[1], self.params[2], self.params[3], y)
        else:
            return NN.h_softmax(x, x.shape[0], self.out_dim, self.h_level1_size, self.h_level2_size, self.params[0], self.params[1], self.params[2], self.params[3])

def concatenate(tensor_list, axis=0):
    """
    Borrow from https://github.com/nyu-dl/dl4mt-tutorial/

    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = T.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = T.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out

#batched dot for tensor3
def batched_dot3(A, B):
    C = A.dimshuffle([0, 1, 2, 'x']) * B.dimshuffle([0, 'x', 1, 2])
    return C.sum(axis=-2)

#batched dot for tensor4
def batched_dot4(A, B):
    C = A.dimshuffle([0, 1, 2, 3, 'x']) * B.dimshuffle([0, 1, 'x', 2, 3])
    return C.sum(axis=-2)

def log_sum_exp():
    pass

def log_softmax(x):
    xdev = x - x.max(1,keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def fast_softmax(x):
    e_x = T.exp(x)
    sm = e_x / e_x.sum(axis=1, keepdims=True)
