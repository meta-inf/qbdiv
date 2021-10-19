import jax
from jax import numpy as np
from jax.scipy.linalg import solve_triangular
import numpy as onp
from utils import *
from kernels import *


class TSLSPredictor(object):

    def __init__(self, beta, kx):
        self.beta, self.kx = beta, kx

    def __call__(self, x_test):
        ret = self.kx.rf_expand(None, None, x_test) @ self.beta
        assert ret.shape[-1] == 1
        return ret, 0 * ret[:, 0]  # mean and diagcov

    def to_onp(self):
        b = self.beta
        self.beta = onp.asarray(b)
        del b

    def to_jnp(self):
        self.beta = np.asarray(b)


def ridge2sls(Dtrain, Dval, lam_space, nu):
    def lin_kfac(x_train):
        mean, sd = x_train.mean(0), x_train.std(0)
        return LinearKernel(inp_stats=(mean, sd), intercept=True)

    Kz, Kx = lin_kfac(Dtrain[0]), lin_kfac(Dtrain[1])
    def proc(dtup):
        z, x, y = dtup
        return Kz.rf_expand(None, None, z), Kx.rf_expand(None, None, x), y

    Dtrain, Dval = tuple(map(proc, (Dtrain, Dval)))
    (Ztr, Xtr, Ytr), (Zva, Xva, Yva) = Dtrain, Dval
    Iz, Ix = np.eye(Ztr.shape[1]), np.eye(Xtr.shape[1])
    beta_yz = np.linalg.solve(Ztr.T @ Ztr + nu * Iz, Ztr.T @ Ytr)
    Ehat = np.linalg.solve(Ztr.T @ Ztr + nu * Iz, Ztr.T @ Xtr)
    PXtr = Ztr @ Ehat  # \hat{E}(x|z=Z) = Z @ Ehat
    PXtr = np.concatenate([PXtr[:, :-1], np.ones_like(PXtr[:, -1:])], -1)
    cov_pxtr = PXtr.T @ PXtr
    ccov_tr = PXtr.T @ Ytr

    best = (1e100, -1)
    for lam in lam_space:
        beta = np.linalg.solve(cov_pxtr + lam * Ix, ccov_tr)
        pred_fva = Xva @ beta
        proj_fva = Zva @ np.linalg.solve(Zva.T @ Zva + nu * Iz, Zva.T @ pred_fva)
        vmse = ((Yva - proj_fva) ** 2).mean()
        # import IPython; IPython.embed(); raise 1
        if not np.isnan(vmse):
            best = min(best, (vmse, lam))

    _, lam = best    
    beta = np.linalg.solve(cov_pxtr + lam * Ix, ccov_tr)
    return TSLSPredictor(beta, Kx), lam

