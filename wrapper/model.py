from layer import get_layer
import numpy as NP
import h5py

class saved_param(object):
    def __init__(self, name, data):
        self.data = data
        self.name = name

    def get_value(self):
        return self.data

    @staticmethod
    def save_list(h5file, theano_data_list):
        params_size = 0
        for tp in theano_data_list:
            params_size += NP.prod(NP.shape(tp.get_value()))
            #print tp.name, NP.shape(tp.get_value())
            h5file[tp.name] = tp.get_value()
        print 'Saving model, total size = ', params_size

    @staticmethod
    def load_file(h5file):
        ret_list = []
        for i in h5file.keys():
            ret_list.append(saved_param(i, h5file[i][:]))

        return ret_list


class LayersPack(object):
    #A helper class to store layers
    def __init__(self):
        self.idxs = {}
    
    def has(self, name):
        return self.idxs.has_key(name)

    def add_instance(self, name, ins):
        assert not self.has(name)
        self.idxs[name] = ins
            
    def get(self, name):
        return self.idxs[name]
    
    def keys(self):
        return self.idxs.keys()
    
    def get_params(self):
        ret_list = []
        for v in self.idxs.values():
            assert hasattr(v, 'get_params')
            ret_list = ret_list + v.get_params()
        return ret_list

    def get_updates(self):
        ret_list = []
        for v in self.idxs.values():
            assert hasattr(v, 'get_updates')
            ret_list = ret_list + v.get_updates()
        return ret_list

    def clear_state(self):
        for v in self.idxs.values():
            assert hasattr(v, 'clear_state')
            v.clear_state()

    def save(self, path):
        theano_data_list = self.get_params()
        h5file = h5py.File(path, 'w')
        saved_param.save_list(h5file, theano_data_list)
        h5file.close()

    def load(self, path):
        h5file = h5py.File(path, 'r')
        params_list = saved_param.load_file(h5file)
        h5file.close()
        for v in self.idxs.values():
            assert hasattr(v, 'load')
            v.load(params_list)

class Model(object):
    def __init__(self):
        self.layersPack = LayersPack()

    def save(self, path):
        self.layersPack.save(path+'.h5')        

    def load(self, path):
        self.layersPack.load(path+'.h5')

    def get_params(self):
        return self.layersPack.get_params()

    def get_updates(self):
        return self.layersPack.get_updates()

    def clear_state(self):
        self.layersPack.clear_state()

    def fc(self, x_in=None, name=None, shape=None, init_list=None):
        return get_layer(self, name=name, layer_type='fc', shape=shape, init_list=init_list).perform(x_in)

    def h_softmax(self, x_in=None, y_in=None, name=None, shape=None, init_list=None):
        return get_layer(self, name=name, layer_type='h_softmax', shape=shape, init_list=init_list).perform(x_in, y_in)

    def gru_seq(self, x_in=None, rec_in=None, name=None, shape=None, init_list=None):
        return get_layer(self, name=name, layer_type='gru_seq', shape=shape, init_list=init_list).perform(x_in, rec_in)
    gru = gru_seq

    def gru_flatten(self, x_in=None, name=None, shape=None, init_list=None, return_seq=True, keep_state=0):
        return get_layer(self, name=name, layer_type='gru_flatten', shape=shape, init_list=init_list, keep_state=keep_state).perform(x_in, return_seq)

    gru_flat = gru_flatten

    def embedding(self, x_in=None, name=None, shape=None, init_list=None):
        return get_layer(self, name=name, layer_type='embedding', shape=shape, init_list=init_list).perform(x_in)
