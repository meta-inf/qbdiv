from typing import Dict, List, Callable, Tuple, Any, Union
from functools import partial
from dataclasses import dataclass

import jax
from jax import numpy as np
import numpy as onp
from flax import struct, optim, linen as nn

import kernels
from utils import *


normal_init = jax.nn.initializers.variance_scaling(2., "fan_in", "normal")


@dataclass
class RFExpander(object):

    n_inp: int
    n_rfs: int
    k: kernels.Kernel
    pkey: np.ndarray

    def __post_init__(self):
        pkey, ckey = jax.random.split(self.pkey)
        self.W = jax.random.normal(ckey, (self.n_inp, self.n_rfs))
        self.b = jax.random.normal(pkey, (self.n_rfs,)) * 2 * np.pi

    def __call__(self, inp: np.ndarray) -> np.ndarray:
        return self.k.rf_expand(self.W, self.b, inp)


class MLP(nn.Module):

    layers: List[int]
    act_type: str = 'elu'
    init_scale: float = 0.9  # ~ prev choice

    def setup(self):
        if hasattr(jax.nn, self.act_type):
            self.activation = getattr(jax.nn, self.act_type)
        elif hasattr(np, self.act_type):
            self.activation = getattr(np, self.act_type)
        else:
            raise NotImplementedError(self.act_type)
        if len(self.layers) == 1:  # RF
            init = normal_init
        else:
            # assuming the input x ~ N(0, I), choose the scale of init var so that 
            # act(Wx/W.shape[0]**0.5) has stddev self.init_scale. These numbers are determined
            # by MC simulation.
            # it's often desirable to use a init_scale < 1. We also need to account for the
            # difference between NTK and NNGP.
            var_scale = {
                'selu': 0.95,
                'relu': 2,
                'elu': 1.5,
                'tanh': 1.5,  # can't do this for tanh as it's bounded, so a random choice here
            }[self.act_type] * self.init_scale**2
            init = jax.nn.initializers.variance_scaling(var_scale, "fan_in", "truncated_normal")
        self.modules = [nn.Dense(features=l, kernel_init=init) for l in self.layers]

    def get_features(self, inp):
        for l in self.modules[:-1]:
            inp = self.activation(l(inp))
        return inp

    def __call__(self, inp, train: bool):
        return self.modules[-1](self.get_features(inp))


class FactorizedEffectNet(nn.Module):

    """
    A potentially more efficient network architecture similar to DFIV, but without BN
    NOTE: we assume the treatment is the last dimension of input, which is true for HLLT and KIV
    """

    ctx_layers: List[int]
    trt_layers: List[int]
    act_type: str
    init_scale: float = 0.9

    def setup(self):
        self.ctx_net = MLP(self.ctx_layers, self.act_type, self.init_scale)
        self.trt_net = MLP(self.trt_layers, self.act_type, self.init_scale)
        self.linear = nn.Dense(features=1)

    def __call__(self, inp: np.ndarray, train: bool) -> np.ndarray:
        bs, d = inp.shape
        ctx, trt = inp[:, :d-1], inp[:, d-1:]
        act = self.ctx_net.activation
        c = act(self.ctx_net(ctx, train=train))[:, None]
        t = act(self.trt_net(trt, train=train))[:, :, None]
        return self.linear((c * t).reshape((bs, -1)))


class Scaled(nn.Module):

    m_fn: Callable[[], nn.Module]
    s: float

    def setup(self):
        self.m = self.m_fn()

    def __call__(self, inp: np.ndarray, train: bool) -> np.ndarray:
        return self.s * self.m(inp, train=train)


class ImageFeatureExtractor(nn.Module):

    gn: bool

    @nn.compact
    def __call__(self, inp, train):
        assert len(inp.shape) == 4 and inp.shape[-1] == 1  # NHWC
        h = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name='conv1')(inp)
        if self.gn: h = nn.GroupNorm(num_groups=8)(h)
        h = nn.relu(h)
        h = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), name='conv2')(inp)
        if self.gn: h = nn.GroupNorm(num_groups=8)(h)
        h = nn.relu(h)
        h = nn.max_pool(h, (2, 2))
        h = h.reshape((h.shape[0], -1))
        h = nn.Dropout(rate=0.05)(h, deterministic=not train)
        h = nn.relu(nn.Dense(features=64)(h))
        h = nn.Dropout(rate=0.05)(h, deterministic=not train)
        return h


class HLLT_MNIST_Model(nn.Module):

    gn: bool
    layers: List[int]
    act_type: str = 'elu'
    init_scale: float = 0.9

    def setup(self):
        self.im_fea = ImageFeatureExtractor(gn=self.gn)
        self.net = MLP(layers=self.layers, act_type=self.act_type, init_scale=self.init_scale)

    def __call__(self, inp, train):
        im_inp = inp[:, 1:-1].reshape((inp.shape[0], 28, 28, 1))
        imf = self.im_fea(im_inp, train)
        h = np.concatenate([inp[:, 0:1], imf, inp[:, -1:None]], 1)
        return self.net(h, train)


@dataclass
class ModifiedRPModel(object):

    f_factory: Callable[[], nn.Module]
    g_factory: Callable[[], nn.Module]
    n_particles: int
    lam: float
    nu: float
    N: int
    x_dims: int
    z_dims: int
    f_rfe: RFExpander
    g_rfe: RFExpander
    f_nn: bool
    g_nn: bool
    pkey0: np.ndarray

    def init_params(self, factory, pkey, inp_dims, scale, nn_correction, n_particles=None):
        n_particles = n_particles or self.n_particles
        inp_shape = (2, inp_dims)
        inp_sample = np.zeros(inp_shape).astype('f')
        net = Scaled(m_fn=factory, s=scale)
        params0 = []
        for _ in range(n_particles):
            pkey, ckey = jax.random.split(pkey)
            params0.append(net.init(ckey, inp_sample, train=False))
        params0_ntk = []
        if nn_correction:
            for _ in range(n_particles):
                pkey, ckey = jax.random.split(pkey)
                params0_ntk.append(net.init(ckey, inp_sample, train=False))
        return net, (params0, params0_ntk)

    def __post_init__(self):
        pkey, ckey = jax.random.split(self.pkey0)
        self.f_net, self.f_params_init = self.init_params(
            self.f_factory, ckey, self.x_dims, 1., self.f_nn)
        self.g_net, self.g_params_init = self.init_params(
            self.g_factory, pkey, self.z_dims, (self.lam/self.nu)**0.5, self.g_nn)

    def __hash__(self):
        return id(self)

    def regularizer(self, prefix, params):
        """
        f (or g, resp.) is initialized as f0=Scaled(f0raw~H, init_sacle) and represented as 
        f=Scaled(fraw, s0).  Here we need
        s_reg * \|f - f0\|_H^2 / 2  =  s_reg * s0**2 * \|f0raw-fraw\|_H^2 / 2.
        """
        ((params0, _), s0, s_reg) = {
            'f': (self.f_params_init, self.f_net.s, self.lam),
            'g': (self.g_params_init, self.g_net.s, self.nu),
        }[prefix]
        raw_sqdist = jax.tree_util.tree_reduce(
            lambda x, y: x+y,
            jax.tree_multimap(lambda p, p0: ((p-p0)**2).sum(), params, params0))
        return raw_sqdist * s_reg * s0**2 / 2

    def be_forward(self, prefix, i, params, inp, train, rng=None):
        """
        prediction of individual BE particles.  When nn_correction is true, we need to add the
        correction -f0 + scale * <p01, \partial f(x;p)/\partial p>, where f0~GP(0, scale**2 NTK).
        As scaling is included in f_net, below is correct
        """
        net = getattr(self, prefix + '_net')
        rng = {'dropout': rng} if rng is not None else None
        ret = net.apply(params, inp, train, rngs=rng)
        if getattr(self, prefix + '_nn'):
            p0s, p01s = getattr(self, prefix + '_params_init')
            p0, p01 = p0s[i], p01s[i]
            # prim_out, tangent_out = jax.jvp(lambda p: net.apply(p, inp), p0, p01)
            prim_out, jvp_fn = jax.linearize(lambda p: net.apply(
                p, inp, train, rngs=rng), p0)  # use the same rng
            tangent_out = jvp_fn(p01)
            ret = ret - prim_out + tangent_out
        return ret

    def loss_fn(
            self,
            all_params: Any,
            dat_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            train: bool,
            rng: Union[np.ndarray, None] = None) -> np.ndarray:
        (z_mb, x_mb, yt_mb, _) = dat_tuple
        z, x = self.g_rfe(z_mb), self.f_rfe(x_mb)
        f_reg = self.regularizer('f', all_params['f'])
        g_reg = self.regularizer('g', all_params['g'])
        loss = (f_reg - g_reg) / self.N
        g_sse = 0
        for i in range(self.n_particles):
            rng, crf, crg = split_pkey(rng, 3)
            f = self.be_forward('f', i, all_params['f'][i], x, train, rng=crf)
            g = self.be_forward('g', i, all_params['g'][i], z, train, rng=crg)
            loss += ((f-yt_mb[:, i])*g - g**2/2).mean()
            g_sse += ((f-yt_mb[:,i]-g)**2).mean()
        stats = {
            'loss': loss,
            'g_mse': g_sse / self.n_particles,
            'g_reg': g_reg,
            'f_reg': f_reg
        }
        return loss, stats

    @partial(jax.jit, static_argnums=(0, 3))
    def _predict(self, params: Any, x_mb: np.ndarray, return_all: bool):
        x = self.f_rfe(x_mb)
        preds = np.stack([
            self.be_forward('f', i, params['f'][i], x, train=False)
            for i in range(self.n_particles)])
        if return_all:
            return np.mean(preds, 0), np.std(preds, 0), preds
        else:
            return np.mean(preds, 0), np.std(preds, 0)

    def predict(self, params: Any, x_mb: np.ndarray, return_all=False):
        return self._predict(params, x_mb, return_all)

    def dump(self):
        td = ['f_rfe', 'g_rfe', 'f_params_init', 'g_params_init']
        return dict((k, getattr(self, k)) for k in td)

    def load(self, dct):
        td = ['f_rfe', 'g_rfe', 'f_params_init', 'g_params_init']
        for k in td:
            object.__setattr__(self, k, dct[k])


class BSModel(ModifiedRPModel):

    def regularizer(self, prefix, params):
        """
        return  s_reg * \|f\|_H^2 / 2 = s_reg * s0**2 * \|fraw\|_H^2 / 2.
        for     f = Scaled(fraw, s0)
        """
        (s0, s_reg) = {
            'f': (self.f_net.s, self.lam),
            'g': (self.g_net.s, self.nu),
        }[prefix]
        raw_sqdist = jax.tree_util.tree_reduce(
            lambda x, y: x+y,
            jax.tree_map(lambda p: (p**2).sum(), params))
        return raw_sqdist * s_reg * s0**2 / 2

    def loss_fn(
            self,
            all_params: Any,
            dat_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            train: bool,
            rng = None) -> np.ndarray:
        (z_mb, x_mb, y_mb, mask_mb) = dat_tuple
        mask_mb = mask_mb[..., None]  #[BS, NP, 1]
        z, x = self.g_rfe(z_mb), self.f_rfe(x_mb)
        f_reg = self.regularizer('f', all_params['f'])
        g_reg = self.regularizer('g', all_params['g'])
        loss = (f_reg - g_reg) / self.N
        g_sse = 0
        for i in range(self.n_particles):
            rng, crf, crg = split_pkey(rng, 3)
            f = self.be_forward('f', i, all_params['f'][i], x, train, rng=crf)
            g = self.be_forward('g', i, all_params['g'][i], z, train, rng=crg)
            loss += (((f-y_mb)*g - g**2/2) * mask_mb[:, i]).sum() / (mask_mb[:, i].sum() + 1e-3)
            g_sse += (((f-y_mb-g)**2) * mask_mb[:, i]).sum() / (mask_mb[:, i].sum() + 1e-3)
        stats = {
            'loss': loss,
            'g_mse': g_sse / self.n_particles,
            'g_reg': g_reg,
            'f_reg': f_reg
        }
        return loss, stats
