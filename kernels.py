import functools
import jax
from jax import numpy as np
import numpy as onp
from utils import *


class Kernel(object):

    def __init__(self):
        pass

    def __call__(self, x1, x2):
        raise NotImplementedError()

    def kdiag(self, x):
        return np.diag(self(x, x))

    def rf_expand(self, W, b, inp):
        # the subsequent layer will handle scaling
        raise NotImplementedError()
        
        
class CircularMaternKernel(Kernel):
    r"""
    Borovitskiy et al (2021), Matern Gaussian processes on Riemannian manifolds. Example 8.
    The spectral density is rho_nu(n) \propto (2nu/kappa^2 + 4pi^2 n^2)^{-nu-1/2}, 
    so we approximate the kernel with truncations
    
    The K-L expansion of the GP is (Eq.(13))
        \sum_n \sqrt{2rho_nu(n)} (eps_{n,1} cos(2pi n x) + eps_{n,2} sin(2pi n x))
    So the kernel is 
        \sum_n 2\rho(n) cos(2pi n x)cos(2pi n x') + sin(2pi n x)sin(2pi n x')
      = \sum_n 2\rho(n) cos(2pi n (x-x'))
    """
    def __init__(self, k, trunc_order=400, kappa=None, x_train=None, var=1.):
        if x_train is None:
            assert kappa is not None
        else:
            assert x_train.shape[-1] == 1
            kappa = median_sqdist(x_train)
        self.kappa = kappa
        self.nu = k/2
        self.var = var
        self.trunc_order = trunc_order
        self.ns = ns = np.arange(trunc_order)
        rho_unnormalized = np.power(2*self.nu/self.kappa**2 + (2*np.pi*ns)**2, -(self.nu+1/2))
        self.rho = rho_unnormalized / rho_unnormalized.sum()
    
    def __call__(self, x1, x2):
        assert len(x1.shape) == len(x2.shape) == 2, 'expect shape [N,1]'
        x_diff = x1[:,None,0] - x2[None,:,0]
        if max(x1.shape[0], x2.shape[0]) <= 500:
            x_diff = x_diff[:, :, None]
            ret = (np.cos(2*np.pi*self.ns[None,None] * x_diff) * self.rho[None,None]).sum(-1)
        else:
            # avoid putting x_diff into JIT buffer
            ret = jax.lax.fori_loop(
                0, self.trunc_order,
                lambda n, cur: (cur[0]+np.cos(2*np.pi*n*cur[1])*self.rho[n], cur[1]),
                (np.zeros(x_diff.shape[:2], dtype=x1.dtype), x_diff)
            )[0]
        return ret * self.var
    
    def kdiag(self, x):
        return np.ones_like(x)[:, 0] * self.var


class LinearKernel(Kernel):
    
    def __init__(self, inp_stats=(0, 1), intercept=False):
        self.inp_stats = inp_stats
        self.intercept = intercept

    def __call__(self, x1, x2):
        t1 = self.rf_expand(None, None, x1)
        t2 = self.rf_expand(None, None, x2)
        return t1 @ t2.T
    
    def kdiag(self, x):
        return (self.rf_expand(None, None, x)**2).sum(axis=-1)

    def rf_expand(self, W, b, inp):
        ret = ((inp - self.inp_stats[0]) / self.inp_stats[1]).astype('f')
        if self.intercept:
            ret = np.concatenate([ret, np.ones_like(ret[:, -1:])], -1)
        return ret


def poly_expand(inp, k):
    assert len(inp.shape) == 2
    n = inp.shape[0]
    # keep it simple
    if k == 2:
        return (inp[:, None] * inp[:, :, None]).reshape(n, -1)
    elif k == 3:
        return (inp[:, None, None, :] * inp[:, None, :, None] * inp[:, :, None, None]).reshape(n, -1)
    assert NotImplementedError(k)


class PolynomialKernel(LinearKernel):

    def __init__(self, x_train, k=2):
        self.k = k
        xf = poly_expand(x_train, k)
        xm, xs = xf.mean(0), xf.std(0)
        super().__init__((xm, xs))

    def rf_expand(self, W, b, inp):
        return super().rf_expand(W, b, poly_expand(inp, self.k))


def median_sqdist(x):
    if x.shape[0] < 3000:
        sqdist = (x[:,None]-x[None,:])**2
        return np.median(sqdist.reshape((x.shape[0]**2, x.shape[-1])), axis=0)
    else:
        ret = []
        for i in range(x.shape[1]):
            ret.append(np.median((x[:,None,i] - x[None,:,i])**2))
        return np.array(ret)


@jax.jit  # let's hope jit creates in-place ops
def get_sqdist(x1, x2, h):
    return (((x1[:,None] - x2[None,:])**2) / h).sum(-1)


class RBFKernel(Kernel):
    
    def __init__(self, var=1., h=None, x_train=None):
        self.var = var
        if h is not None:
            self.h = h
        else:
            self.h = median_sqdist(x_train[:2500])
    
    def __call__(self, x1, x2):
        return self.var * np.exp(-get_sqdist(x1, x2, self.h) / 2)
    
    def kdiag(self, x):
        return self.var * np.ones((x.shape[0],))

    def rf_expand(self, W, b, inp):
        inp = inp / self.h**0.5
        return (self.var**0.5 * np.cos(inp @ W + b)).astype('f')


class ScaleMixtureKernel(Kernel):

    def __init__(self, x_train, scales=[0.1, 1, 10], KBase=RBFKernel, **kw):
        h = median_sqdist(x_train)
        self.ks = []
        for s in scales:
            kw['h'] = h * s
            self.ks.append(KBase(**kw))

    @functools.partial(jax.jit, static_argnums=(0,))
    def __call__(self, x1, x2):
        ret = self.ks[0](x1, x2)
        for k in self.ks[1:]:
            ret += k(x1, x2)
        return ret / len(self.ks)

    def kdiag(self, x):
        return sum(k.kdiag(x) for k in self.ks) / len(self.ks)

    def rf_expand(self, W, b, inp):
        return (sum(k.rf_expand(W, b, inp) for k in self.ks) / len(self.ks)**0.5).astype('f')

    
class MaternKernel(Kernel):
    
    def __init__(self, k, var=1., h=None, x_train=None):
        self.k = k
        self.var = var
        assert k in [1, 3, 5, 7]
        if h is not None:
            self.h = h
        else:
            assert x_train is not None
            sqdist = (x_train[:,None]-x_train[None,:])**2
            self.h = np.median(sqdist.reshape((x_train.shape[0]**2, x_train.shape[-1])), axis=0)

    def __call__(self, x1, x2):
        dist = ((x1[:,None] - x2[None,:])**2 / self.h).sum(-1) ** 0.5
        d = (self.k**0.5) * dist
        if self.k == 1:
            ret = np.exp(-d)
        elif self.k == 3:
            ret = (1 + d) * np.exp(-d)
        elif self.k == 5:
            ret = (1 + d + d**2/3) * np.exp(-d)
        elif self.k == 7:
            ret = (1 + d + 2/5*d**2 + d**3/15) * np.exp(-d)
        return self.var * ret
    
    def kdiag(self, x):
        return self.var * np.ones((x.shape[0],))
