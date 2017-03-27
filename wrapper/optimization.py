import theano
import theano.tensor as T
import numpy as NP

def pure_sgd(cost, params, lr, rescale=1., ignore_input_disconnect=False, return_norm=False):  #lr and iterations must be theano variable
    grads = theano.grad(cost, params, disconnected_inputs='ignore' if ignore_input_disconnect else 'raise')
    grad_norm = T.sqrt(sum(map(lambda x:T.sqr(x).sum(), grads)))

    updates = []

    new_norm = 0
    for p,g in zip(params, grads):
        g = g * (rescale/T.maximum(rescale, grad_norm))
        new_norm = new_norm + T.sqr(g).sum()
        new_p = p - lr * g
        updates.append((p, new_p))
    if return_norm:
        return updates, grad_norm

    return updates

#need fix
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

def rmsprop(cost, params, lr=0.0001, rho=0.99, epsilon=1e-6, rescale=1. , ignore_input_disconnect=False, return_norm=False):
    '''
    Borrowed from keras, no constraints, though
    '''
    updates = []

    if isinstance(lr, list):
        lrs=lr
    else:
        lrs=[lr]*len(params)
    if isinstance(epsilon, list):
        epsilons=epsilon
    else:
        epsilons=[epsilon]*len(params)

    grads = theano.grad(cost, params, disconnected_inputs='ignore' if ignore_input_disconnect else 'raise')
    grad_norm = T.sqrt(sum(map(lambda x:T.sqr(x).sum(), grads)))
    acc = [theano.shared(NP.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in params]
    for p, g, a, lr, epsilon in zip(params, grads, acc, lrs, epsilons):
        g = g * (rescale/T.maximum(rescale, T.sqrt(grad_norm)))
        new_a = rho * a + (1 - rho) * g ** 2
        updates.append((a, new_a))
        new_p = p - lr * g / T.sqrt(new_a + epsilon)
        updates.append((p, new_p))
    if return_norm:
        return updates, grad_norm
    return updates

def Adam(cost, params, lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, rescale=1., decay=1e-4, ignore_input_disconnect=False, return_norm=False):
    iters = theano.shared(NP.cast[theano.config.floatX](0))

    updates = []
    grads = theano.grad(cost, params, disconnected_inputs='ignore' if ignore_input_disconnect else 'raise')
    grad_norm = T.sqrt(sum(map(lambda x:T.sqr(x).sum(), grads)))
    updates.append((iters, iters+1))
    t = iters+1
    lr = lr * (1./(1.+decay*iters)) * T.sqrt(1-beta_2**t)/(1-beta_1**t)

    for p, g in zip(params, grads):
        g = g * (rescale/T.maximum(rescale, T.sqrt(grad_norm)))
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)

        m_t = (beta_1 * m) + (1 - beta_1) * g
        v_t = (beta_2 * v) + (1 - beta_2) * (g**2)
        p_t = p - lr * m_t / (T.sqrt(v_t) + epsilon)

        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    if return_norm:
        return updates, grad_norm
    return updates
