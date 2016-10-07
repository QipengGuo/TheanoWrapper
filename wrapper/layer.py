import theano
import theano.tensor as T
import numpy as NP
from basic_ops import Linear, LinearLN, Fetch, Conv, Pooling, H_Softmax, concatenate, batched_dot3, batched_dot4, log_sum_exp, log_softmax, fast_softmax
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
    def __init__(self, model, name=None, layer_type='fully_connect', shape=[], init_list=None, linear_mode=None, with_bias=True):
        super(self.__class__, self).__init__(model, name=name, layer_type=layer_type)
	_Linear = LinearLN if linear_mode=='LN' else Linear
        if not self.created:
            self.ops.append(_Linear(name=name+'_Linear', shape=shape, init_list=init_list if init_list is not None, with_bias=with_bias))

    def perform(self, x):
        L_trans = self.ops[0].p
        return L_trans(x)

class h_softmax(Layer):
    def __init__(self, model, name=None, layer_type='h_softmax', shape=[], init_list=None):
        super(self.__class__, self).__init__(model, name=name, layer_type=layer_type)
        if not self.created:
            self.ops.append(H_Softmax(name=name+'_H_Softmax', shape=shape, init_list=init_list if init_list is not None))

    def perform(self, x, y=None):
        return self.ops[0].p(x, y)

class lstm_seq(Layer):
    def __init__(self, model, name=None, layer_type='lstm_seq', shape=[], init_list=None, init_list_inner=None, f_act=None, f_inner_act=None, linear_mode=None):
        assert len(shape)==2
        super(self.__class__, self).__init__(model, name=name, layer_type=layer_type)
        _Linear = LinearLN if linear_mode=='LN' else Linear
        self.f_act = tanh if f_act is None else f_act
        self.f_inner_act = sigmoid if f_inner_act is None else f_inner_act
        self.name=name
        in_dim, out_dim = shape
        self.in_dim, self.out_dim = in_dim, out_dim
        if not self.created:
            self.ops.append(_Linear(name=name+'_Linear_i_Wb', shape=(in_dim, out_dim), init_list=init_list if init_list is not None else ['glorot_uniform', 'zeros']))
            self.ops.append(_Linear(name=name+'_Linear_i_U', shape=(out_dim, out_dim), init_list=init_list_inner if init_list_inner is not None else ['orthogonal', 'zeros'], with_bias=False))
            self.ops.append(_Linear(name=name+'_Linear_o_Wb', shape=(in_dim, out_dim), init_list=init_list if init_list is not None else ['glorot_uniform', 'zeros']))
            self.ops.append(_Linear(name=name+'_Linear_o_U', shape=(out_dim, out_dim), init_list=init_list_inner if init_list_inner is not None else ['orthogonal', 'zeros'], with_bias=False))
            self.ops.append(_Linear(name=name+'_Linear_f_Wb', shape=(in_dim, out_dim), init_list=init_list if init_list is not None else ['glorot_uniform', 'zeros']))
            self.ops.append(_Linear(name=name+'_Linear_f_U', shape=(out_dim, out_dim), init_list=init_list_inner if init_list_inner is not None else ['orthogonal', 'zeros'], with_bias=False))
            self.ops.append(_Linear(name=name+'_Linear_c_Wb', shape=(in_dim, out_dim), init_list=init_list if init_list is not None else ['glorot_uniform', 'zeros']))
            self.ops.append(_Linear(name=name+'_Linear_c_U', shape=(out_dim, out_dim), init_list=init_list_inner if init_list_inner is not None else ['orthogonal', 'zeros'], with_bias=False))


    def perform(self, x, cm1):
        ops = self.ops
        if hasattr(self.f_inner_act, '__iter__'):
            i_t = self.f_inner_act[0](ops[0].p(x)+ops[1].p(cm1))
            o_t = self.f_inner_act[1](ops[2].p(x)+ops[3].p(cm1))
            f_t = self.f_inner_act[2](ops[4].p(x)+ops[5].p(cm1))
        else:
            i_t = self.f_inner_act(ops[0].p(x)+ops[1].p(cm1))
            o_t = self.f_inner_act(ops[2].p(x)+ops[3].p(cm1))
            f_t = self.f_inner_act(ops[4].p(x)+ops[5].p(cm1))
        ct_t = self.f_act(ops[6].p(x)+ops[7].p(cm1))
        c_t = f_t * cm1 + i_t * ct_t
        h_t = o_t * self.f_act(c_t)

        return c_t, h_t

class gru_seq(Layer):
    def __init__(self, model, name=None, layer_type='gru_seq', shape=[], init_list=None, init_list_inner=None, f_act=None, f_inner_act=None, linear_mode=None):
        assert len(shape)==2
        super(self.__class__, self).__init__(model, name=name, layer_type=layer_type)
	_Linear = LinearLN if linear_mode=='LN' else Linear
	self.f_act = tanh if f_act is None else f_act
	self.f_inner_act = sigmoid if f_inner_act is None else f_inner_act
        in_dim, out_dim = shape
        if not self.created:
            self.ops.append(_Linear(name=name+'_Linear_z_Wb', shape=(in_dim, out_dim), init_list=init_list if init_list is not None else ['glorot_uniform', 'zeros']))
            self.ops.append(_Linear(name=name+'_Linear_z_U', shape=(out_dim, out_dim), init_list=init_list_inner if init_list_inner is not None else ['orthogonal', 'zeros'], with_bias=False))
            self.ops.append(_Linear(name=name+'_Linear_r_Wb', shape=(in_dim, out_dim), init_list=init_list if init_list is not None else ['glorot_uniform', 'zeros']))
            self.ops.append(_Linear(name=name+'_Linear_r_U', shape=(out_dim, out_dim), init_list=init_list_inner if init_list_inner is not None else ['orthogonal', 'zeros'], with_bias=False))
            self.ops.append(_Linear(name=name+'_Linear_h_Wb', shape=(in_dim, out_dim), init_list=init_list if init_list is not None else ['glorot_uniform', 'zeros']))
            self.ops.append(_Linear(name=name+'_Linear_h_U', shape=(out_dim, out_dim), init_list=init_list_inner if init_list_inner is not None else ['orthogonal', 'zeros'], with_bias=False))

    def perform(self, x, hm1):
        ops = self.ops
        if hasattr(self.f_inner_act, '__iter__'):
            z_t = self.f_inner_act[0](ops[0].p(x) + ops[1].p(hm1))
            r_t = self.f_inner_act[1](ops[2].p(x) + ops[3].p(hm1))
        else:
            z_t = self.f_inner_act(ops[0].p(x) + ops[1].p(hm1))
            r_t = self.f_inner_act(ops[2].p(x) + ops[3].p(hm1))
        h_hat_t = self.f_act(ops[4].p(x) + ops[5].p(hm1*r_t))
        h_t = (1-z_t) * hm1 + z_t * h_hat_t
        return h_t

class lstm_flatten(Layer):
    def __init__(self, model, name=None, layer_type='lstm_flatten', shape=[], init_list=None, init_list_inner=None, keep_state=0, f_act=None, f_inner_act=None, linear_mode=None):
        assert len(shape)==2
        super(self.__class__, self).__init__(model, name=name, layer_type=layer_type)
        _Linear = LinearLN if linear_mode=='LN' else Linear
        self.f_act = tanh if f_act is None else f_act
        self.f_inner_act = sigmoid if f_inner_act is None else f_inner_act
        self.name=name
        in_dim, out_dim = shape
        self.in_dim, self.out_dim = in_dim, out_dim
        self.state = None
        self.keep_state=keep_state
        if not self.created:
            self.ops.append(_Linear(name=name+'_Linear_i_Wb', shape=(in_dim, out_dim), init_list=init_list if init_list is not None else ['glorot_uniform', 'zeros']))
            self.ops.append(_Linear(name=name+'_Linear_i_U', shape=(out_dim, out_dim), init_list=init_list_inner if init_list_inner is not None else ['orthogonal', 'zeros'], with_bias=False))
            self.ops.append(_Linear(name=name+'_Linear_o_Wb', shape=(in_dim, out_dim), init_list=init_list if init_list is not None else ['glorot_uniform', 'zeros']))
            self.ops.append(_Linear(name=name+'_Linear_o_U', shape=(out_dim, out_dim), init_list=init_list_inner if init_list_inner is not None else ['orthogonal', 'zeros'], with_bias=False))
            self.ops.append(_Linear(name=name+'_Linear_f_Wb', shape=(in_dim, out_dim), init_list=init_list if init_list is not None else ['glorot_uniform', 'zeros']))
            self.ops.append(_Linear(name=name+'_Linear_f_U', shape=(out_dim, out_dim), init_list=init_list_inner if init_list_inner is not None else ['orthogonal', 'zeros'], with_bias=False))
            self.ops.append(_Linear(name=name+'_Linear_c_Wb', shape=(in_dim, out_dim), init_list=init_list if init_list is not None else ['glorot_uniform', 'zeros']))
            self.ops.append(_Linear(name=name+'_Linear_c_U', shape=(out_dim, out_dim), init_list=init_list_inner if init_list_inner is not None else ['orthogonal', 'zeros'], with_bias=False))


    def step(self, i_wx, o_wx, f_wx, c_wx, cm1, hm1):
        ops = self.ops
        if hasattr(self.f_inner_act, '__iter__'):
            i_t = self.f_inner_act[0](i_wx + ops[1].p(hm1))
            o_t = self.f_inner_act[1](o_wx + ops[3].p(hm1))
            f_t = self.f_inner_act[2](f_wx + ops[5].p(hm1))
        else:
            i_t = self.f_inner_act(i_wx + ops[1].p(hm1))
            o_t = self.f_inner_act(o_wx + ops[3].p(hm1))
            f_t = self.f_inner_act(f_wx + ops[5].p(hm1))
        ct_t = self.f_act(c_wx + ops[7].p(hm1))
        c_t = f_t * cm1 + i_t * ct_t
        h_t = o_t * tanh(c_t)
        return c_t, h_t

    def perform(self, x, return_seq=True):
        if self.state is None and self.keep_state:
            self.state = theano.shared(NP.zeros((self.keep_state, self.out_dim), dtype=theano.config.floatX))
        state = self.state if self.keep_state else T.zeros((x.shape[0], self.out_dim))
        ops = self.ops
        flat_x = x.reshape((-1, x.shape[2]))
        i_wx = ops[0].p(flat_x).reshape((x.shape[0], x.shape[1], -1))
        o_wx = ops[2].p(flat_x).reshape((x.shape[0], x.shape[1], -1))
        f_wx = ops[4].p(flat_x).reshape((x.shape[0], x.shape[1], -1))
        c_wx = ops[6].p(flat_x).reshape((x.shape[0], x.shape[1], -1))

        sc, _ = theano.scan(self.step, sequences=[i_wx.dimshuffle(1,0,2), o_wx.dimshuffle(1,0,2), f_wx.dimshuffle(1,0,2), c_wx.dimshuffle(1,0,2)], outputs_info=[state, state], name=self.name+'_scan')

        if self.keep_state:
            self.updates = [(self.state, sc[-1])]
        if return_seq:
            return sc[1].dimshuffle(1,0,2)
        else:
            return sc[1][-1]

class gru_flatten(Layer):
    def __init__(self, model, name=None, layer_type='gru_flatten', shape=[], init_list=None, init_list_inner=None, keep_state=0, f_act=None, f_inner_act=None, linear_mode=None):
        assert len(shape)==2
        super(self.__class__, self).__init__(model, name=name, layer_type=layer_type)
	_Linear = LinearLN if linear_mode=='LN' else Linear
	self.f_act = tanh if f_act is None else f_act
	self.f_inner_act = sigmoid if f_inner_act is None else f_inner_act
        self.name=name
        in_dim, out_dim = shape
        self.in_dim, self.out_dim = in_dim, out_dim
        self.state = None
        self.keep_state=keep_state
        if not self.created:
            self.ops.append(_Linear(name=name+'_Linear_z_Wb', shape=(in_dim, out_dim), init_list=init_list if init_list is not None else ['glorot_uniform', 'zeros']))
            self.ops.append(_Linear(name=name+'_Linear_z_U', shape=(out_dim, out_dim), init_list=init_list_inner if init_list_inner is not None else ['orthogonal', 'zeros'], with_bias=False))
            self.ops.append(_Linear(name=name+'_Linear_r_Wb', shape=(in_dim, out_dim), init_list=init_list if init_list is not None else ['glorot_uniform', 'zeros']))
            self.ops.append(_Linear(name=name+'_Linear_r_U', shape=(out_dim, out_dim), init_list=init_list_inner if init_list_inner is not None else ['orthogonal', 'zeros'], with_bias=False))
            self.ops.append(_Linear(name=name+'_Linear_h_Wb', shape=(in_dim, out_dim), init_list=init_list if init_list is not None else ['glorot_uniform', 'zeros']))
            self.ops.append(_Linear(name=name+'_Linear_h_U', shape=(out_dim, out_dim), init_list=init_list_inner if init_list_inner is not None else ['orthogonal', 'zeros'], with_bias=False))

    def step(self, z_wx, r_wx, h_hat_wx, hm1):
        ops = self.ops
        if hasattr(self.f_inner_act, '__iter__'):
            z_t = self.f_inner_act[0](z_wx + ops[1].p(hm1))
            r_t = self.f_inner_act[1](r_wx + ops[3].p(hm1))
        else:
            z_t = self.f_inner_act(z_wx + ops[1].p(hm1))
            r_t = self.f_inner_act(r_wx + ops[3].p(hm1))
        h_hat_t = self.f_act(h_hat_wx + ops[5].p(hm1*r_t))
        h_t = (1-z_t) * hm1 + z_t * h_hat_t
        return h_t

    def perform(self, x, return_seq=True):
        if self.state is None and self.keep_state:
            self.state = theano.shared(NP.zeros((self.keep_state, self.out_dim), dtype=theano.config.floatX))
        state = self.state if self.keep_state else T.zeros((x.shape[0], self.out_dim))
        ops = self.ops
        flat_x = x.reshape((-1, x.shape[2]))
        z_wx = ops[0].p(flat_x).reshape((x.shape[0], x.shape[1], -1))
        r_wx = ops[2].p(flat_x).reshape((x.shape[0], x.shape[1], -1))
        h_hat_wx = ops[4].p(flat_x).reshape((x.shape[0], x.shape[1], -1))
        sc, _ = theano.scan(self.step, sequences=[z_wx.dimshuffle(1,0,2), r_wx.dimshuffle(1,0,2), h_hat_wx.dimshuffle(1,0,2)], outputs_info=[state], name=self.name+'_scan')
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
            self.ops.append(Fetch(name=name+'_Emb_Dict', shape=shape, init_list=init_list if init_list is not None else ['uniform']))

    def perform(self, x):
        return self.ops[0].p(x)
        

class wmatrix(Layer):
    def __init__(self, model, name=None, layer_type='Wmatrix', shape=[], init_list=None):
        super(self.__class__, self).__init__(model, name=name, layer_type=layer_type)
        if not self.created: 
            self.ops.append(Fetch(name=name+'_Wmatrix', shape=shape, init_list=init_list if init_list is not None else ['uniform']))

    def perform(self):
        return self.ops[0].p(get_all=True)

class attention(Layer):
    def __init__(self, model, name=None, layer_type='attention', shape=[], init_list=None, linear_mode=None):
        super(self.__class__, self).__init__(model, name=name, layer_type=layer_type)
	_Linear = LinearLN if linear_mode=='LN' else Linear
        if not self.created:
            x_dim, h_dim = shape
            self.ops.append(_Linear(name=name+'_Linear_W', shape=(x_dim, h_dim), init_list=init_list if init_list is not None))
            self.ops.append(_Linear(name=name+'_Linear_U', shape=(h_dim, h_dim), init_list=init_list if init_list is not None, with_bias=False))
            self.ops.append(_Linear(name=name+'_Linear_V', shape=(h_dim, 1), init_list=init_list if init_list is not None, with_bias=False))

                
    def perform(self, x, h):
        Wx_b = self.ops[0].p(x)
        Uh = self.ops[1].p(h)
	alpha = (Wx_b[:,None,:]+Uh[None,:,:]).reshape((x.shape[0]*h.shape[0],h.shape[1]))
	alpha = self.ops[2].p(tanh(alpha)).reshape((x.shape[0],h.shape[0]))
        return softmax_fast(alpha)

class conv(Layer):
    def __init__(self, model, name=None, layer_type='conv', shape=[], init_list=None):
        super(self.__class__, self).__init__(model, name=name, layer_type=layer_type)
        if not self.created: 
            self.ops.append(Conv(name=name+'_conv', shape=shape, init_list=init_list if init_list is not None))

    def perform(self, x):
        return self.ops[0].p(x)

class pooling(Layer):
    def __init__(self, model, name=None, layer_type='pooling', shape=[]):
        super(self.__class__, self).__init__(model, name=name, layer_type=layer_type)
	self.ops.append(Pooling(name=name+'_Pooling', shape=shape))

    def perform(self, x):
        return self.ops[0].p(x)

