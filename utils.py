# import torch as th
from __future__ import annotations
from typing import Iterable, Callable, Any, Union, Tuple
import numpy as onp
import scipy
import jax
from jax import numpy as np


class UniformNystromSampler(object):
    """
    the sampler will not modify its state 
    """
    def __init__(self, pkey: jax.random.PRNGKey, rho=0.5, a=1, mn=60):
        self.pkey, self.rho, self.a, self.mn = pkey, rho, a, mn
       
    def __call__(self, z_all: np.ndarray) -> np.ndarray:
        n = z_all.shape[0]
        m = min(max(self.mn, int(self.a * n**self.rho)), n)
        idcs = jax.random.permutation(self.pkey, n)[:m]
        return z_all[idcs]


def split_pkey(k: Union[jax.random.PRNGKey, None], num: int = 2):
    if k is not None:
        return jax.random.split(k, num)
    return tuple([None] * num)


class PRNGKeyHolder(object):

    """ For use inside a function. """

    def __init__(self, pkey):
        self.pkey = pkey

    def gen_key(self):
        self.pkey, ckey = jax.random.split(self.pkey)
        return ckey


def ceil_div(a, b):  return (a + b - 1) // b


def gen_bs_mask(pkey, N, ratio, n_particles):
    if ratio + 1e-4 < 1:
        Mk = np.ones((N, n_particles), dtype=np.bool_)
        idcs = np.arange(N)
        bs_n_removed = max(int(N * (1 - ratio)), 1)
        for i in range(n_particles):
            pkey, ckey = jax.random.split(pkey)
            excluded = jax.random.choice(ckey, idcs, (bs_n_removed, 1), replace=False)
            Mk = jax.ops.index_update(Mk, jax.ops.index[excluded, i], False)
    else:
        Mk = np.zeros((N, n_particles), dtype=np.float32)
        for i in range(n_particles):
            pkey, ckey = jax.random.split(pkey)
            idcs  = jax.random.randint(ckey, (N, 1), 0, N)
            Mk = jax.ops.index_add(Mk, jax.ops.index[idcs, i], 1)
    return Mk


def l2_regularizer(params: Any) -> np.ndarray:
    return jax.tree_util.tree_reduce(
        lambda x, y: x+y,
        jax.tree_map(lambda p: (p**2).sum(), params))


def ci_coverage(
        actual: np.ndarray, pmean: np.ndarray, psd: np.ndarray, r: float = 0.95) -> np.ndarray:
    assert actual.shape == pmean.shape == psd.shape
    k = scipy.stats.norm.ppf((1+r) / 2)
    return np.logical_and(pmean - k*psd <= actual, actual <= pmean + k*psd).mean()


def mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == b.shape
    return ((a-b) ** 2).mean()


def normal_loglh(mean, sd, val):
    return -0.5 * (np.log(np.pi*2) + 2*np.log(sd) + ((mean-val)/sd)**2)


class TensorDataLoader(object):
    """ Lightweight DataLoader . TensorDataset for jax """

    def __init__(self, *arrs, batch_size=None, shuffle=False, rng=None, dtype=np.float32):
        assert batch_size is not None
        self.arrs = [a.astype(dtype) for a in arrs]
        self.N, self.B = arrs[0].shape[0], batch_size
        self.shuffle = shuffle
        assert all(a.shape[0] == self.N for a in arrs[1:])
        self.rng = rng if rng is not None else onp.random.RandomState(23)

    def __iter__(self):
        idcs = onp.arange(self.N)
        if self.shuffle:
            self.rng.shuffle(idcs)
        self.arrs_cur = [a[idcs] for a in self.arrs]
        self.i = 0
        return self

    def __next__(self):
        if self.i < self.N:
            old_i = self.i
            self.i = min(self.i + self.B, self.N)
            return tuple(a[old_i:self.i] for a in self.arrs)
        else:
            raise StopIteration

    def __len__(self):
        return (self.N+self.B-1) // self.B


def split_list_args(s, typ=float):
    if len(s.strip()) == 0:
        return []
    return list(map(typ, s.split(',')))


def data_split(*arrs, split_ratio=0.8, rng=None):
    if rng == None:
        rng = onp.random
    N = arrs[0].shape[0]
    assert all(a.shape[0] == N for a in arrs)
    idcs = onp.arange(N)
    rng.shuffle(idcs)
    split = int(N * split_ratio)
    train_tuple = tuple(a[idcs[:split]] for a in arrs)
    test_tuple = tuple(a[idcs[split:]] for a in arrs)
    return train_tuple, test_tuple


def log_linspace(s, e, n):
    return onp.exp(onp.linspace(onp.log(s), onp.log(e), n))


class Accumulator(object):
    
    def __init__(self):
        self.a = []
        
    def append(self, d):
        # if isinstance(d, th.Tensor):
        #     d = d.item()
        if isinstance(d, jax.numpy.ndarray) and hasattr(d, '_device'):
            d = float(d)
        self.a.append(d)
        
    def average(self):
        return onp.mean(self.a)
    
    def minimum(self, s=0):
        return onp.min(self.a[s:])

    def maximum(self, s=0):
        return onp.max(self.a[s:])

    def argmin(self):
        return onp.argmin(self.a)

    def __getitem__(self, i):
        return self.a[i]


class StatsDict(dict):

    def __init__(self, *args):
        if len(args) == 0:
            super().__init__()
        else:
            assert len(args) == 1
            a = args[0]
            if isinstance(a, dict):
                super().__init__(a.items())
            else:
                super().__init__(a)

    def add_prefix(self, pref: str, sep: str = '/') -> StatsDict:
        return StatsDict((pref + sep + k, v) for k, v in self.items())

    def filter(self, pred_or_key: Union[Callable[[str], True], str]):
        pred = pred_or_key if callable(pred_or_key) else lambda k: k == pred_or_key
        return StatsDict((k, v) for k, v in self.items() if not pred(k))


class StatsAccumulator(object):

    def __init__(self):
        pass

    def append(self, d: dict):
        if not hasattr(self, 'stats'):
            self.stats = dict((k, Accumulator()) for k in d)
        for k in d:
            v = float(d[k])
            self.stats[k].append(v)

    def dump(self) -> StatsDict:
        return StatsDict((k, self.stats[k].average()) for k in self.stats)

    def __getitem__(self, k: str) -> Accumulator:
        return self.stats[k]


def traverse_ds(
        step_fn: Callable[[Any], Any], dset: Iterable[Any], has_stats: bool,
        rng: Union[np.ndarray, None] = None
    ) -> Tuple[float, StatsDict]:
    stats = StatsAccumulator()
    for data in dset:
        if rng is not None:
            rng, c_rng = jax.random.split(rng)
            ret = step_fn(data, rng=c_rng)
        else:
            ret = step_fn(data)
        if has_stats:
            loss, rdct = ret
            rdct['_loss'] = loss
        else:
            rdct = StatsDict({'_loss': ret})
        stats.append(rdct)
    ret = stats.dump()
    return ret['_loss'], ret


class DummyContext(object):
    
    def __init__(self, v):
        self.v = v
    
    def __enter__(self):
        return self.v
    
    def __exit__(self, *args, **kw):
        pass
    
    def set_postfix(self):
        pass


def add_bool_flag(parser, name, default=None):
    parser.add_argument('-'+name, action='store_true', default=default)
    parser.add_argument('-no_'+name, action='store_false', dest=name)
