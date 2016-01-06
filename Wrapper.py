import theano
import theano.tensor as T
import theano.tensor.nnet as NN
import numpy as NP
import numpy.random as RNG
#import h5py 
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

### Utility functions begin
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
    for i in xrange(len(args[0])/batch_size):
        for j in xrange(len(args)):
            targs[j] = args[j][i*batch_size:(i+1)*batch_size]
        t = func(*targs)
        if len(t)>1:
            if i==0:
                result = [None]*len(t)
                for j in xrange(len(t)):
                    result[j] = []
            for j in xrange(len(t)):
                result[j].append(t[j])
        else:
            result.append(t)
    if len(t)>1:
        for i in xrange(len(t)):
            result[i]=NP.asarray(result[i])
            if result[i].ndim>1:
                result[i] = NP.concatenate(result[i], axis=0)
        return result
    else:
        result = NP.asarray(result)
        if result.ndim>1:
            return NP.concatenate(result, axis=0)
        else:
            return result
    return None

### Utility functions end

class Dropout(object):
	def __init__(self, shape = None, prob=0.5):
		self.seed = RNG.randint(1e6)
		self.retain_prob = 1.0 - prob
		self.rng = RandomStreams(seed=self.seed)
		self.mask = self.rng.binomial(shape, p=self.retain_prob)

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
                conv2d = CUDNN.dnn_conv

    def save(self, path):
        self.weightsPack.save(path+'_W.npz')
        self.layersPack.save(path+'_L.npz')

    def load(self, path):
        self.weightsPack.load(path+'_W.npz')
        self.layersPack.load(path+'_L.npz')

    def clear(self):
        self.weightsPack = WeightsPack()
        self.layersPack = LayersPack()

    def gru(self, cur_in=None, rec_in=None, name=None, shape=[]):  
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


    def att_mem(self, cur_in = None, mem_in = None, name = None, shape=[], tick= None):
            if len(shape)<1 and (name in self.layersPack.keys()):
                    shape = layersPack.get(name)

            cur_dim, rec_dim = shape
            h_dim = rec_dim
            params = [None]*3
            Wname_list = [name+'_W', name+'_U', name+'_V']
            if name not in self.layersPack.keys():
                    #W
                    params[0] = theano.shared(glorot_uniform((cur_dim, h_dim)), name='W_att')
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
            mem = mem_in[:tick+1].dimshuffle([0, 1, 2, 'x'])

            Wx_Uh = T.dot(cur_in, params[0]).dimshuffle(['x', 0, 1, 'x']) + batched_dot4(params[1].dimshuffle(['x', 'x', 0, 1]), mem)
            att = batched_dot4(params[2].dimshuffle(['x', 'x', 0, 1]), T.tanh(Wx_Uh))
            att = T.extra_ops.squeeze(T.patternbroadcast(att, (False,False,True,True)))
            return NN.softmax(att)

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

def rmsprop(cost, params, lr=0.0005, rho=0.9, epsilon=1e-6):
    '''
    Borrowed from keras, no constraints, though
    '''
    updates = OrderedDict()
    grads = theano.grad(cost, params)
    acc = [theano.shared(NP.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in params]
    for p, g, a in zip(params, grads, acc):
        new_a = rho * a + (1 - rho) * g ** 2
        updates[a] = new_a
        new_p = p - lr * g / T.sqrt(new_a + epsilon)
        updates[p] = new_p

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
        self.idxs = data['idxs'].tolist()
        self.num_elem = data['num_elem'].tolist()
        self.vect = data['vect'].tolist()

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
    
