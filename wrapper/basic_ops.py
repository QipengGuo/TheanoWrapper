import numpy as NP
import numpy.random as RNG
import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import initialization as INIT
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
if theano.config.device[:3]=='cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
if theano.config.device[:3]=='gpu':
    from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams

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


class Stuff_list(object):
    init_name_list = {'uniform':INIT.uniform, 'glorot_uniform':INIT.glorot_uniform, 'orthogonal':INIT.orthogonal}
    @staticmethod
    def get_init(init_name):
        assert Stuff_list.init_name_list.has_key(init_name)
        return Stuff_list.init_name_list[init_name]

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
            self.params[i] = theano.shared(Stuff_list.get_init(init_list[i])(shape_list[i]), name=name_list[i])
        self.save_weights=save_weights
        self.created = True

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
            for p in data:
                if p.name == target_name:
                    self.params[i].set_value(p.get_value())
                    finds += 1
                    break
        return finds == len(self.params)
'''
#Do we really need this?

def Ops_without_weights(object):
    def __init__(self, name = None):
        self.op_name = op_name
        self.created = False

    def create(self):
        if self.created:
            return 
        return NotImplemented

    def perform(self):
        return NotImplemented

    def get_updates(self):
        return None

    def get_params(self):
        return None
'''

class Linear(Ops_with_weights):
    def __init__(self, name=None, save_weights=True, shape=[], with_bias=True, init_list=['glorot_uniform', 'glorot_uniform']):
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

class Conv(Ops_with_weights):
    pass

class Pool(object):
    pass

class Fetch(Ops_with_weights):
    def __init__(self, name=None, save_weights=True, shape=[], init_list=['uniform']):
        assert len(init_list)==1
        assert len(shape)==2
        super(self.__class__, self).__init__(name)

        in_dim, out_dim = shape
        shape_list = [(in_dim, out_dim)]
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
    def __init__(self, name=None, save_weights=True, shape=[], init_list=['glorot_uniform', 'glorot_uniform', 'glorot_uniform', 'glorot_uniform']):
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

def log_softmax():
    pass

def fast_softmax():
    pass



