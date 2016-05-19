import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import theano.tensor.extra_ops as Tex
import numpy as NP
import numpy.random as RNG
#import h5py 
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

### Utility functions begin

def softmax(x):
    e_x = T.exp(x)
    sm = e_x / T.sum(e_x, axis=0, keepdims=True)
    return sm

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

def merge_OD(A, B):
	C = OrderedDict()
	for k,e in A.items()+B.items():
		C[k]=e
	return C

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
    return NP.cast[T.config.floatX](RNG.uniform(low=-s, high=s, size=shape))

def orthogonal(shape, scale=1.1):
    '''
    Borrowed from keras
    '''
    flat_shape = (shape[0], NP.prod(shape[1:]))
    a = RNG.normal(0, 1, flat_shape)
    u, _, v = NP.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return NP.cast[T.config.floatX](q)

#batched dot for tensor3
def batched_dot3(A, B):
	C = A.dimshuffle([0, 1, 2, 'x']) * B.dimshuffle([0, 'x', 1, 2])
	return C.sum(axis=-2)

#batched dot for tensor4
def batched_dot4(A, B):
	C = A.dimshuffle([0, 1, 2, 3, 'x']) * B.dimshuffle([0, 1, 'x', 2, 3])
	return C.sum(axis=-2)

def auto_batch(func, batch_size, *args):
    result = []
    t=None
    targs = [None]*len(args)
    for i in xrange(len(args[0])/batch_size+1):
        for j in xrange(len(args)):
            targs[j] = args[j][i*batch_size:(i+1)*batch_size]
        if i*batch_size>=len(args[0]):
            break
        t = func(*targs)
        if len(t)>1 and not isinstance(t, NP.ndarray):
            if i==0:
                result = [None]*len(t)
                for j in xrange(len(t)):
                    result[j] = []
            for j in xrange(len(t)):
                result[j].append(t[j])
        else:
            result.append(t)
    if len(t)>1 and not isinstance(t, NP.ndarray):
        for i in xrange(len(t)):
            result[i]=NP.asarray(result[i])
            if isinstance(result[i][0], (list, tuple, NP.ndarray)):
                result[i] = NP.concatenate(result[i], axis=0)
        return result
    else:
        result = NP.asarray(result)
        if isinstance(result[0], (list, tuple, NP.ndarray)):
            return NP.concatenate(result, axis=0)
        else:
            return result
    return None

### Utility functions end

class Dropout(object):
	def __init__(self, shape = None, prob=0.5):
		self.seed = RNG.randint(1e6)
		self.retain_prob = 1.0 - prob
		self.shape = shape
		self.rng = RandomStreams(seed=self.seed)
		self.mask = self.rng.binomial(shape, p=self.retain_prob, dtype=theano.config.floatX)

	def drop(self, cur_in):
		h = cur_in * self.mask
		h /= self.retain_prob
		return h

class Model(object):
    def __init__(self):
        self.weightsPack = WeightsPack()
        self.layersPack = LayersPack()
        self.conv2d = NN.conv2d
        if theano.config.device[:3] == 'gpu':
            import theano.sandbox.cuda.dnn as CUDNN
            if CUDNN.dnn_available():
                print 'Using CUDNN instead of Theano conv2d'
                self.conv2d = CUDNN.dnn_conv

    def save(self, path):
        self.weightsPack.save(path+'_W.npz')
        self.layersPack.save(path+'_L.npz')

    def load(self, path):
        self.weightsPack.load(path+'_W.npz')
        self.layersPack.load(path+'_L.npz')

    def clear(self):
        self.weightsPack = WeightsPack()
        self.layersPack = LayersPack()

    def lstm(self, cur_in=None, rec_in=None, rec_mem = None, name=None, shape=[]):  
        if len(shape)<1 and (name in self.layersPack.keys()):
            shape = layersPack.get(name)

        in_dim, out_dim = shape
        params = [None]*12
        Wname_list = [name+'_W_h', name+'_U_h', name+'_b_h',name+'_W_i', name+'_U_i', name+'_b_i', name+'_W_o', name+'_U_o', name+'_b_o', name+'_W_f', name+'_U_f', name+'_b_f']
        if name not in self.layersPack.keys():
            #W_h, U_h, b_h
            params[0] = theano.shared(glorot_uniform((in_dim, out_dim)), name = 'W_h')
            params[1] = theano.shared(orthogonal((out_dim, out_dim)), name = 'U_h')
            params[2] = theano.shared(NP.zeros((out_dim,), dtype=theano.config.floatX), name='b_h')
            #W_i, U_i, b_i
            params[3] = theano.shared(glorot_uniform((in_dim, out_dim)), name = 'W_i')
            params[4] = theano.shared(orthogonal((out_dim, out_dim)), name = 'U_i')
            params[5] = theano.shared(NP.zeros((out_dim,), dtype=theano.config.floatX), name='b_i')
            #W_o, U_o, b_o
            params[6] = theano.shared(glorot_uniform((in_dim, out_dim)), name = 'W_o')
            params[7] = theano.shared(orthogonal((out_dim, out_dim)), name = 'U_o')
            params[8] = theano.shared(NP.zeros((out_dim,), dtype=theano.config.floatX), name='b_o')
            #W_f, U_f, b_f
            params[9] = theano.shared(glorot_uniform((in_dim, out_dim)), name = 'W_f')
            params[10] = theano.shared(orthogonal((out_dim, out_dim)), name = 'U_f')
            params[11] = theano.shared(NP.zeros((out_dim,), dtype=theano.config.floatX), name='b_f')

            #add W to weights pack
            self.weightsPack.add_list(params, Wname_list)

            #add gru to layers pack
            self.layersPack.add(name, shape, ltype='lstm')
        else:
            for i in xrange(len(Wname_list)):
                params[i] = self.weightsPack.get(Wname_list[i])

	_h_t = T.tanh(T.dot(cur_in, params[0]) + T.dot(rec_in, params[1]) + params[2])
	i_t = NN.sigmoid(T.dot(cur_in, params[3]) + T.dot(rec_in, params[4]) + params[5])
	o_t = NN.sigmoid(T.dot(cur_in, params[6]) + T.dot(rec_in, params[7]) + params[8])
	f_t = NN.sigmoid(T.dot(cur_in, params[9]) + T.dot(rec_in, params[10]) + params[11]) 
        c_t = i_t * _h_t + f_t * rec_mem
	lstm_t = T.tanh(c_t) * o_t

        return lstm_t, c_t

    def gru(self, cur_in=None, rec_in=None, name=None, shape=[]):  
        if len(shape)<1 and (name in self.layersPack.keys()):
            shape = layersPack.get(name)

        in_dim, out_dim = shape
        params = [None]*3
        Wname_list = [name+'_W', name+'_U', name+'_b']
        if name not in self.layersPack.keys():
            #W_h, U_h, b_h
            params[0] = theano.shared(NP.concatenate((glorot_uniform((in_dim, out_dim)), glorot_uniform((in_dim, out_dim)), glorot_uniform((in_dim, out_dim))), axis=1), name = 'W')
            params[1] = theano.shared(NP.concatenate((orthogonal((out_dim, out_dim)), orthogonal((out_dim, out_dim)),orthogonal((out_dim, out_dim))), axis=1), name = 'U')
            params[2] = theano.shared(NP.concatenate((NP.zeros((out_dim,), dtype=theano.config.floatX), NP.zeros((out_dim,), dtype=theano.config.floatX), NP.zeros((out_dim,), dtype=theano.config.floatX)), axis=0), name = 'b')

            #add W to weights pack
            self.weightsPack.add_list(params, Wname_list)

            #add gru to layers pack
            self.layersPack.add(name, shape, ltype='gru')
        else:
            for i in xrange(len(Wname_list)):
                params[i] = self.weightsPack.get(Wname_list[i])

        Wx = T.dot(cur_in, params[0])
        gates = NN.sigmoid(Wx[:,:2*out_dim]+T.dot(rec_in, params[1][:,:2*out_dim])+params[2][:2*out_dim]) # 0:out_dim r, out_dim:2*out_dim z
        _gru_h = T.tanh(Wx[:,2*out_dim:]+T.dot(rec_in * gates[:,:out_dim], params[1][:,2*out_dim:])+params[2][2*out_dim:])
        gru_h = (1-gates[:,out_dim:]) * rec_in + gates[:,out_dim:] * _gru_h
        
        return gru_h

    def gru_bak(self, cur_in=None, rec_in=None, name=None, shape=[]):  
        if len(shape)<1 and (name in self.layersPack.keys()):
            shape = layersPack.get(name)

        in_dim, out_dim = shape
        params = [None]*9
        Wname_list = [name+'_W_h', name+'_U_h', name+'_b_h',name+'_W_r', name+'_U_r', name+'_b_r', name+'_W_z', name+'_U_z', name+'_b_z']
        if name not in self.layersPack.keys():
            #W_h, U_h, b_h
            params[0] = theano.shared(glorot_uniform((in_dim, out_dim)), name = 'W_h')
            params[1] = theano.shared(orthogonal((out_dim, out_dim)), name = 'U_h')
            params[2] = theano.shared(NP.zeros((out_dim,), dtype=theano.config.floatX), name='b_h')
            #W_r, U_r, b_r
            params[3] = theano.shared(glorot_uniform((in_dim, out_dim)), name = 'W_r')
            params[4] = theano.shared(orthogonal((out_dim, out_dim)), name = 'U_r')
            params[5] = theano.shared(NP.zeros((out_dim,), dtype=theano.config.floatX), name='b_r')
            #W_z, U_z, b_z
            params[6] = theano.shared(glorot_uniform((in_dim, out_dim)), name = 'W_z')
            params[7] = theano.shared(orthogonal((out_dim, out_dim)), name = 'U_z')
            params[8] = theano.shared(NP.zeros((out_dim,), dtype=theano.config.floatX), name='b_z')

            #add W to weights pack
            self.weightsPack.add_list(params, Wname_list)

            #add gru to layers pack
            self.layersPack.add(name, shape, ltype='gru')
        else:
            for i in xrange(len(Wname_list)):
                params[i] = self.weightsPack.get(Wname_list[i])

        gru_r = NN.sigmoid(T.dot(cur_in, params[3]) + T.dot(rec_in, params[4]) + params[5])
        gru_z = NN.sigmoid(T.dot(cur_in, params[6]) + T.dot(rec_in, params[7]) + params[8])
        _gru_h = T.tanh(T.dot(cur_in, params[0]) + T.dot(rec_in * gru_r, params[1]) + params[2])
        gru_h = (1 - gru_z) * rec_in + gru_z * _gru_h

        return gru_h
           
    def fc(self, cur_in=None, name=None, shape=[]):
        if len(shape)<1 and (name in self.layersPack.keys()):
            shape = layersPack.get(name)

        in_dim, out_dim = shape
        params = [None]*2
        Wname_list = [name+'_W',name+'_b']
        if name not in self.layersPack.keys():
            #W
            params[0] = theano.shared(glorot_uniform((in_dim, out_dim)), name='W_h')
            #b
            params[1] = theano.shared(NP.zeros((out_dim,), dtype=theano.config.floatX), name='b_h')

            #add W to weights pack
            self.weightsPack.add_list(params, Wname_list)

            #add fc to layers pack
            self.layersPack.add(name, shape, ltype='fc')
        else:
            for i in xrange(len(Wname_list)):
                params[i] = self.weightsPack.get(Wname_list[i])

        fc_h = T.dot(cur_in, params[0])+params[1]

        return fc_h

    def embedding(self, cur_in = None, name= None, shape=[]):
        if len(shape)<1 and (name in self.layersPack.keys()):
            shape = layersPack.get(name)

        max_idx, emb_size = shape
        max_idx += 1
        params = [None]
        Wname_list = [name+'_DICT']
        if name not in self.layersPack.keys():
            #DICT
            params[0] = theano.shared(glorot_uniform((max_idx, emb_size)))

            #add W to weights pack
            self.weightsPack.add_list(params, Wname_list)
            #add fc to layers pack
            self.layersPack.add(name, shape, ltype='embedding')
        else:
            for i in xrange(len(Wname_list)):
                params[i] = self.weightsPack.get(Wname_list[i])
        #one_hot = Tex.to_one_hot(cur_in, max_idx)
        
        #emb = T.dot(one_hot, params[0])
        emb = params[0][cur_in]
        return emb

    def Wmatrix(self, name= None, shape=[]):
        if len(shape)<1 and (name in self.layersPack.keys()):
            shape = layersPack.get(name)

        params = [None]
        Wname_list = [name+'_Wmatrix']
        if name not in self.layersPack.keys():
            #DICT
            params[0] = theano.shared(orthogonal(shape), name='W_matrix')

            #add W to weights pack
            self.weightsPack.add_list(params, Wname_list)
            #add fc to layers pack
            self.layersPack.add(name, shape, ltype='Wmatrix')
        else:
            for i in xrange(len(Wname_list)):
                params[i] = self.weightsPack.get(Wname_list[i])
        return params[0]

    def att_mem_2in(self, cur_in1 = None, cur_in2 = None, mem_in = None, name = None, shape=[], act_func=None):
            if act_func is None:
                act_func = softmax
            if len(shape)<1 and (name in self.layersPack.keys()):
                    shape = layersPack.get(name)

            cur1_dim, cur2_dim, rec_dim = shape
            h_dim = rec_dim
            params = [None]*4
            Wname_list = [name+'_W1', name+'_W2', name+'_U', name+'_V']
            if name not in self.layersPack.keys():
                    #W1
                    params[0] = theano.shared(glorot_uniform((cur1_dim, h_dim)), name='W1_att')
                    params[1] = theano.shared(glorot_uniform((cur2_dim, h_dim)), name='W2_att')
                    #U
                    params[2] = theano.shared(orthogonal((rec_dim, h_dim)), name='U_att')
                    #V
                    params[3] = theano.shared(glorot_uniform((1, h_dim)), name='V_att')
                    
                    self.weightsPack.add_list(params, Wname_list)

                    self.layersPack.add(name, shape, ltype='att_mem')
            else:
                    for i in xrange(len(Wname_list)):
                            params[i] = self.weightsPack.get(Wname_list[i])

            #time, batch, channel, 1
            mem = mem_in.dimshuffle([0, 1, 2, 'x'])
            Wx_Uh = T.dot(cur_in1, params[0]).dimshuffle(['x', 0, 1, 'x']) + T.dot(cur_in2, params[1]).dimshuffle(['x', 0, 1, 'x']) + batched_dot4(params[2].dimshuffle(['x', 'x', 0, 1]), mem)
            v_t = params[3]
            att = batched_dot4(v_t.dimshuffle(['x', 'x', 0, 1]), T.tanh(Wx_Uh))
            att = att[:,:,0,0] 
            return act_func(att)

    def att_mem(self, cur_in = None, mem_in = None, name = None, shape=[], act_func=None):
            if act_func is None:
                act_func = softmax
            if len(shape)<1 and (name in self.layersPack.keys()):
                    shape = layersPack.get(name)

            cur_dim, rec_dim = shape
            h_dim = rec_dim
            params = [None]*3
            Wname_list = [name+'_W', name+'_U', name+'_V']
            if name not in self.layersPack.keys():
                    #W1
                    params[0] = theano.shared(glorot_uniform((cur_dim, h_dim)), name='W1_att')
                    #U
                    params[1] = theano.shared(orthogonal((rec_dim, h_dim)), name='U_att')
                    #V
                    params[2] = theano.shared(glorot_uniform((1, h_dim)), name='V_att')
                    
                    self.weightsPack.add_list(params, Wname_list)

                    self.layersPack.add(name, shape, ltype='att_mem')
            else:
                    for i in xrange(len(Wname_list)):
                            params[i] = self.weightsPack.get(Wname_list[i])

            #time, batch, channel, 1
            mem = mem_in.dimshuffle([0, 1, 2, 'x'])
            Wx_Uh = T.dot(cur_in, params[0]).dimshuffle(['x', 0, 1, 'x']) + batched_dot4(params[1].dimshuffle(['x', 'x', 0, 1]), mem)
            v_t = params[2]
            att = batched_dot4(v_t.dimshuffle(['x', 'x', 0, 1]), T.tanh(Wx_Uh))
            att = att[:,:,0,0] 
            return act_func(att)


    def conv(self, cur_in=None, name=None, shape=[]):
        if len(shape)<1 and (name in self.layersPack.keys()):
            shape = layersPack.get(name)

        nr_filters, nr_channels, filter_row, filter_col, conv_stride_row, conv_stride_col = shape
        params = [None] * 1
        Wname_list = [name+'_W']
        if name not in self.layersPack.keys():
            #W
            params[0] = theano.shared(glorot_uniform((nr_filters, nr_channels, filter_row, filter_col)), name='conv_W') 

            #add W to weighs pack
            self.weightsPack.add_list(params, Wname_list)

            #add conv to layers pack
            self.layersPack.add(name, shape, ltype='conv')
        else:
            for i in xrange(len(Wname_list)):
                params[i] = self.weightsPack.get(Wname_list[i])
        
        conv_h = self.conv2d(cur_in, params[0], subsample=(conv_stride_row, conv_stride_col)) 

        return conv_h
        
def sgd(cost, params, lr, iterations, momentum=0.9, decay=0.05):  #lr and iterations must be theano variable
    grads = theano.grad(cost, params)
    lr *= (1.0 / (1.0 + decay * iterations))

    updates = []

    updates.append((iterations, iterations + 1.))
    for p,g in zip(params, grads):
        m = theano.shared(NP.zeros(p.get_value().shape, dtype=theano.config.floatX))
        v = momentum * m - lr * g
        updates.append((m, v))

        new_p = p + momentum * v - lr * g
        updates.append((p, new_p))
    return updates

def rmsprop(cost, params, lr=0.0001, rho=0.99, epsilon=1e-6, rescale=1. , ignore_input_disconnect=False):
    '''
    Borrowed from keras, no constraints, though
    '''
    updates = OrderedDict()
    grads = theano.grad(cost, params, disconnected_inputs='ignore' if ignore_input_disconnect else 'raise')
    grad_norm = T.sqrt(sum(map(lambda x:T.sqr(x).sum(), grads)))
    acc = [theano.shared(NP.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in params]
    for p, g, a in zip(params, grads, acc):
	g = g * (rescale/T.maximum(rescale, T.sqrt(grad_norm)))
        new_a = rho * a + (1 - rho) * g ** 2
        updates[a] = new_a
        new_p = p - lr * g / T.sqrt(new_a + epsilon)
        updates[p] = new_p

    return updates
class Adam(object):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsi = epsilon
        self.iters = theano.shared(NP.cast[theano.config.floatX](0))

    def get_updates(self, cost, params):
        updates = OrderedDict()
        grads = theano.grad(cost, params)
        updates[self.iters] = self.iters+1
        t = self.iters+1
        lr = self.lr * T.sqrt(1-self.beta_2**t)/(1-self.beta_1**t)

        for p, g in zip(params, grads):
            m = theano.shared(p.get_value() * 0.)
            v = theano.shared(p.get_value() * 0.)

            m_t = (self.beta_1 * m) + (1 - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1 - self.beta_2) * (g**2)
            p_t = p - lr * m_t / (T.sqrt(v_t) + self.epsi)

            updates[m] = m_t
            updates[v] = v_t
            updates[p] = p_t
        return updates
        
class WeightsPack(object):
    #A helper class to index theano variables into a pack.
    def __init__(self):
        self.idxs = {}
        self.num_elem = 0
        self.vect =[]

    def add(self, data, name):
        self.idxs[name] = self.num_elem
        self.num_elem += 1
        self.vect.append(data)

    def get(self, name):
        return self.vect[self.idxs[name]]
    
    def add_list(self, data_list, name_list):
        for i in xrange(len(data_list)):
            self.add(data_list[i], name_list[i])
            
    def getW_list(self):
        return self.vect
    
    def save(self, path):
        NP.savez(path, idxs=self.idxs, num_elem=self.num_elem, vect=self.vect)

        #file = h5py.File(path, 'w')
        #file.create_dataset('W_idxs', data=(self.idxs))
        #file.create_dataset('W_num_elem', data=(self.num_elem))
        #file.create_dataset('W_vect', data=(self.vect))
        #file.close()
        
    def load(self, path):
        data = NP.load(path)
        idxs = data['idxs'].tolist()
        num_elem = data['num_elem'].tolist()
        vect = data['vect'].tolist()
        if len(idxs) == len(self.idxs):
            for i in idxs.keys():
                self.vect[self.idxs[i]].set_value(vect[idxs[i]].get_value())
        else:
            print 'Create weights'
            print idxs
            print self.idxs
            print '-------------------------'
            self.idxs = idxs
            self.num_elem = num_elem
            self.vect = vect
        #file = h5py.File(path, 'r')
        #self.idxs  = file['W_idxs'][:]
        #self.num_elem = file['W_num_elem'][:]
        #self.vect = file['W_vect'][:]
        #file.close()


class LayersPack(object):
    #A helper class to store layers
    def __init__(self):
        self.idxs = {}
        self.num_elem = 0
        self.shape = []
    
    def add(self, name, shape, ltype):
        self.idxs[name] = self.num_elem
        self.num_elem += 1
        self.shape.append(shape)
            
    def get(self, name):
        return self.shape[self.idxs[name]]
    
    def keys(self):
        return self.idxs.keys()
    
    def save(self, path):
        NP.savez(path, idxs=self.idxs, num_elem=self.num_elem, shape=self.shape)
        #file = h5py.File(path, 'w')
        #file.create_dataset('L_idxs', data=(self.idxs))
        #file.create_dataset('L_num_elem', data=(self.num_elem))
        #file.create_dataset('L_shape', data=(self.shape))
        #file.close()

    def load(self, path):
        data = NP.load(path)
        self.idxs = data['idxs'].tolist()
        self.num_elem = data['num_elem'].tolist()
        self.shape = data['shape'].tolist()
        #file = h5py.File(path, 'r')
        #self.idxs  = file['L_idxs'][:]
        #self.num_elem = file['L_num_elem'][:]
        #self.shape = file['L_shape'][:]
        #file.close()
    
