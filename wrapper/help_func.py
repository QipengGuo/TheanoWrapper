import theano.tensor as T
import numpy as NP


def letanh(x):
    return 1.7159*tanh(2.0/3.0*x)

def categorical_crossentropy(prob, true_idx):
    true_idx = T.arange(true_idx.shape[0]) * prob.shape[1] + true_idx
    t1 = prob.flatten()[true_idx]
    return -t1

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
