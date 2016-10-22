import theano
import theano.tensor as T
import numpy as NP

def pure_sgd(cost, params, lr, rescale=1., ignore_input_disconnect=False):  #lr and iterations must be theano variable
    grads = theano.grad(cost, params, disconnected_inputs='ignore' if ignore_input_disconnect else 'raise')
    grad_norm = T.sqrt(sum(map(lambda x:T.sqr(x).sum(), grads)))

    updates = []

    new_norm = 0
    for p,g in zip(params, grads):
        g = g * (rescale/T.maximum(rescale, grad_norm))
        new_norm = new_norm + T.sqr(g).sum()
        new_p = p - lr * g
        updates.append((p, new_p))
    return updates

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

def rmsprop(cost, params, lr=0.0001, rho=0.99, epsilon=1e-6, rescale=1. , ignore_input_disconnect=False, return_norm=True):
    '''
    Borrowed from keras, no constraints, though
    '''
    updates = []

    grads = theano.grad(cost, params, disconnected_inputs='ignore' if ignore_input_disconnect else 'raise')
    grad_norm = T.sqrt(sum(map(lambda x:T.sqr(x).sum(), grads)))
    acc = [theano.shared(NP.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in params]
    for p, g, a in zip(params, grads, acc):
        g = g * (rescale/T.maximum(rescale, T.sqrt(grad_norm)))
        new_a = rho * a + (1 - rho) * g ** 2
        updates.append((a, new_a))
        new_p = p - lr * g / T.sqrt(new_a + epsilon)
        updates.append((p, new_p))
	if return_norm:
		return updates, grad_norm
    return updates

#need fix
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
