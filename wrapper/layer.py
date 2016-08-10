import theano
import theano.tensor as T
import numpy as NP
from basic_ops import Linear, Fetch, Conv, Pool, H_Softmax, concatenate, batched_dot3, batched_dot4, log_sum_exp, log_softmax, fast_softmax
from activation import sigmoid, tanh, softmax, relu, softmax_fast, relu_leak
class Layer(object):
    def __init__(self, model, name=None, layer_type=None):
        assert hasattr(model, 'layersPack')
        assert name is not None and layer_type is not None
        self.model = model
        self.layer_name = name
        self.layer_type = layer_type
        self.created = False
        self.ops = []
        self.updates = []
        if not model.layersPack.has(name):
            model.layersPack.add_instance(name, self)
        else:
            self.created = True

    def perform(self):
        return NotImplemented

    def load(self, data):
        for op in self.ops:
            assert hasattr(op, 'load')
            op.load(data)

    def clear_state(self):
        for update in self.updates:
            update[0].set_value(update[0].get_value()*0.0)

    def get_updates(self):
        ret = self.updates
        for op in self.ops:
            assert hasattr(op, 'get_updates')
            ret = ret + op.get_updates()
        return ret

    def get_params(self):
        ret = []
        for op in self.ops:
            assert hasattr(op, 'get_params')
            ret = ret + op.get_params()
        return ret


class fully_connect(Layer):
    def __init__(self, model, name=None, layer_type='fully_connect', shape=[], init_list=None):
        super(self.__class__, self).__init__(model, name=name, layer_type=layer_type)
        if not self.created:
            if init_list is not None:
                self.ops.append(Linear(name=name+'_Linear', shape=shape, init_list=init_list))
            else:
                self.ops.append(Linear(name=name+'_Linear', shape=shape))

    def perform(self, x):
        L_trans = self.ops[0].p
        return L_trans(x)

class h_softmax(Layer):
    def __init__(self, model, name=None, layer_type='h_softmax', shape=[], init_list=None):
        super(self.__class__, self).__init__(model, name=name, layer_type=layer_type)
        if not self.created:
            if init_list is not None:
                self.ops.append(H_Softmax(name=name+'_H_Softmax', shape=shape, init_list=init_list))
            else:
                self.ops.append(H_Softmax(name=name+'_H_Softmax', shape=shape))

    def perform(self, x, y=None):
        return self.ops[0].p(x, y)

class lstm_seq(Layer):
    pass

class gru_seq(Layer):
    def __init__(self, model, name=None, layer_type='gru_seq', shape=[], init_list=None, init_list_inner=None):
        assert len(shape)==2
        super(self.__class__, self).__init__(model, name=name, layer_type=layer_type)
        in_dim, out_dim = shape
        if not self.created:
            if init_list is not None:
                assert init_list_inner is not None
                self.ops.append(Linear(name=name+'_Linear_z_Wb', shape=(in_dim, out_dim), init_list=init_list))
                self.ops.append(Linear(name=name+'_Linear_z_U', shape=(out_dim, out_dim), init_list=init_list_inner, with_bias=False))
                self.ops.append(Linear(name=name+'_Linear_r_Wb', shape=(in_dim, out_dim), init_list=init_list))
                self.ops.append(Linear(name=name+'_Linear_r_U', shape=(out_dim, out_dim), init_list=init_list_inner, with_bias=False))
                self.ops.append(Linear(name=name+'_Linear_h_Wb', shape=(in_dim, out_dim), init_list=init_list))
                self.ops.append(Linear(name=name+'_Linear_h_U', shape=(out_dim, out_dim), init_list=init_list_inner, with_bias=False))
            else:
                self.ops.append(Linear(name=name+'_Linear_z_Wb', shape=(in_dim, out_dim), init_list=['glorot_uniform', 'glorot_uniform']))
                self.ops.append(Linear(name=name+'_Linear_z_U', shape=(out_dim, out_dim), init_list=['orthogonal', 'glorot_uniform'], with_bias=False))
                self.ops.append(Linear(name=name+'_Linear_r_Wb', shape=(in_dim, out_dim), init_list=['glorot_uniform', 'glorot_uniform']))
                self.ops.append(Linear(name=name+'_Linear_r_U', shape=(out_dim, out_dim), init_list=['orthogonal', 'glorot_uniform'], with_bias=False))
                self.ops.append(Linear(name=name+'_Linear_h_Wb', shape=(in_dim, out_dim), init_list=['glorot_uniform', 'glorot_uniform']))
                self.ops.append(Linear(name=name+'_Linear_h_U', shape=(out_dim, out_dim), init_list=['orthogonal', 'glorot_uniform'], with_bias=False))

    def perform(self, x, hm1):
        #z_W = params[0], z_b = params[1], z_U=params[2], r_W = params[3], r_b = params[4], r_U = params[5], h_W = params[6], h_b = params[7], h_U = params[8]
        ops = self.ops
        z_t = sigmoid(ops[0].p(x) + ops[1].p(hm1))
        r_t = sigmoid(ops[2].p(x) + ops[3].p(hm1))
        h_hat_t = tanh(ops[4].p(x) + ops[5].p(hm1*r_t))
        h_t = (1-z_t) * hm1 + z_t * h_hat_t
        return h_t

class lstm_flatten(Layer):
    pass

class gru_flatten(Layer):
    def __init__(self, model, name=None, layer_type='gru_flatten', shape=[], init_list=None, init_list_inner=None, keep_state=0):
        assert len(shape)==2
        super(self.__class__, self).__init__(model, name=name, layer_type=layer_type)
        self.name=name
        in_dim, out_dim = shape
        self.in_dim, self.out_dim = in_dim, out_dim
        self.state = None
        self.keep_state=keep_state
        if not self.created:
            if init_list is not None:
                assert init_list_inner is not None
                self.ops.append(Linear(name=name+'_Linear_z_Wb', shape=(in_dim, out_dim), init_list=init_list))
                self.ops.append(Linear(name=name+'_Linear_z_U', shape=(out_dim, out_dim), init_list=init_list_inner, with_bias=False))
                self.ops.append(Linear(name=name+'_Linear_r_Wb', shape=(in_dim, out_dim), init_list=init_list))
                self.ops.append(Linear(name=name+'_Linear_r_U', shape=(out_dim, out_dim), init_list=init_list_inner, with_bias=False))
                self.ops.append(Linear(name=name+'_Linear_h_Wb', shape=(in_dim, out_dim), init_list=init_list))
                self.ops.append(Linear(name=name+'_Linear_h_U', shape=(out_dim, out_dim), init_list=init_list_inner, with_bias=False))
            else:
                self.ops.append(Linear(name=name+'_Linear_z_Wb', shape=(in_dim, out_dim), init_list=['glorot_uniform', 'glorot_uniform']))
                self.ops.append(Linear(name=name+'_Linear_z_U', shape=(out_dim, out_dim), init_list=['orthogonal', 'glorot_uniform'], with_bias=False))
                self.ops.append(Linear(name=name+'_Linear_r_Wb', shape=(in_dim, out_dim), init_list=['glorot_uniform', 'glorot_uniform']))
                self.ops.append(Linear(name=name+'_Linear_r_U', shape=(out_dim, out_dim), init_list=['orthogonal', 'glorot_uniform'], with_bias=False))
                self.ops.append(Linear(name=name+'_Linear_h_Wb', shape=(in_dim, out_dim), init_list=['glorot_uniform', 'glorot_uniform']))
                self.ops.append(Linear(name=name+'_Linear_h_U', shape=(out_dim, out_dim), init_list=['orthogonal', 'glorot_uniform'], with_bias=False))

    def step(self, z_wx, r_wx, h_hat_wx, hm1):
        ops = self.ops
        z_t = sigmoid(z_wx + ops[1].p(hm1))
        r_t = sigmoid(r_wx + ops[3].p(hm1))
        h_hat_t = tanh(h_hat_wx + ops[5].p(hm1*r_t))
        h_t = (1-z_t) * hm1 + z_t * h_hat_t
        return h_t

    def perform(self, x, return_seq=True):
        #z_W = params[0], z_b = params[1], z_U=params[2], r_W = params[3], r_b = params[4], r_U = params[5], h_W = params[6], h_b = params[7], h_U = params[8]
        if self.state is None and self.keep_state:
            self.state = theano.shared(NP.zeros((self.keep_state, self.out_dim), dtype=theano.config.floatX))
        state = self.state if self.keep_state else T.zeros((x.shape[0], self.out_dim))
        ops = self.ops
        flat_x = x.reshape((-1, x.shape[2]))
        z_wx = ops[0].p(flat_x).reshape((x.shape[0], x.shape[1], -1))
        r_wx = ops[2].p(flat_x).reshape((x.shape[0], x.shape[1], -1))
        h_hat_wx = ops[4].p(flat_x).reshape((x.shape[0], x.shape[1], -1))
        sc, _ = theano.scan(self.step, sequences=[z_wx.dimshuffle(1,0,2), r_wx.dimshuffle(1,0,2), h_hat_wx.dimshuffle(1,0,2)], outputs_info=[state], name=self.name+'_scan')
        #if len(self.updates)<1:
        if self.keep_state:
            self.updates = [(self.state, sc[-1])]
        if return_seq:
            return sc.dimshuffle(1,0,2)
        else:
            return sc[-1]

class embedding(Layer):
    def __init__(self, model, name=None, layer_type='embedding', shape=[], init_list=None):
        super(self.__class__, self).__init__(model, name=name, layer_type=layer_type)
        if not self.created:
            if init_list is not None:
                self.ops.append(Fetch(name=name+'_Emb_Dict', shape=shape, init_list=init_list))
            else:
                self.ops.append(Fetch(name=name+'_Emb_Dict', shape=shape, init_list=['uniform']))

    def perform(self, x):
        return self.ops[0].p(x)
        

class attention(Layer):
    pass

class conv(Layer):
    pass

class pool(Layer):
    pass

def get_layer(model, name=None, layer_type='', **kwargs):
    ltype_dict = {'fully_connect':fully_connect, 'fc':fully_connect, 'lstm':lstm_seq, 'lstm_seq':lstm_seq, \
        'lstm_flatten':lstm_flatten, 'gru_seq':gru_seq, 'gru_flatten':gru_flatten, 'gru':gru_seq, 'embedding':embedding, \
        'emb':embedding, 'conv':conv, 'pool':pool, 'h_softmax':h_softmax}
    assert ltype_dict.has_key(layer_type)
    assert hasattr(model, 'layersPack')
    if model.layersPack.has(name):
        return model.layersPack.get(name)
    else:
        print kwargs
        return ltype_dict[layer_type](model, name=name, layer_type=layer_type, **kwargs) #really?
