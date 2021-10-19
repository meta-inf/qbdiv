from __future__ import annotations
from typing import Union

import jax
from jax import numpy as np
from jax.scipy.linalg import solve_triangular
import numpy as onp

from utils import *
from kernels import *


def _matmul_with_Kxy(A: np.ndarray, Kx: Kernel, X: np.ndarray,
                     Y: Union[np.ndarray, None] = None,
                     blk_size=5000) -> np.ndarray:
    """ evaluate A @ Kxy without constructing the latter """
    if Y is None:
        Y = X
    n = X.shape[0]
    def step_fn(t):
        i, prev = t
        A_slice = jax.lax.dynamic_slice(A, (0, i), (A.shape[0], blk_size))
        K_slice = Kx(jax.lax.dynamic_slice(X, (i, 0), (blk_size, X.shape[1])), Y)
        return i+blk_size, prev+A_slice@K_slice
    A_Kxx = 0
    li = n // blk_size * blk_size
    if n >= blk_size:
        R0 = np.zeros((A.shape[0], Y.shape[0]), dtype=A.dtype)
        A_Kxx += jax.lax.while_loop(lambda t: t[0]<li, step_fn, (0, R0))[1]
    if n % blk_size != 0:
        A_Kxx += A[:, li: li+blk_size] @ Kx(X[li: li+blk_size], Y)
    return A_Kxx


class LRLinearMap(object):
    """
    represents A @ B
    """
    def __init__(self, A, B):
        self.A, self.B = (A, B)

    def __call__(self, b: np.ndarray) -> np.ndarray:
        assert len(b.shape) == 2, b.shape
        if self.B is None:
            return self.A @ b
        return self.A @ (self.B @ b)

    def to_onp(self):
        for k, v in vars(self).items():
            if isinstance(v, np.ndarray):
                setattr(self, k, onp.asarray(v))
                del v  # np arrays have a default ref count of 2

    def to_jnp(self):
        for k, v in vars(self).items():
            if isinstance(v, onp.ndarray):
                setattr(self, k, np.asarray(v))

    def __sub__(self, b):
        return LRLinearMap(self.A-b.A, self.B-b.B)

    def _matmul_Kxy(self, Kx: Kernel, X: np.ndarray, Y=None) -> LRLinearMap:
        if Y is None:
            Y = X
        if self.B is None:
            return LRLinearMap(self.A @ Kx(X, Y), None)
        return LRLinearMap(self.A, _matmul_with_Kxy(self.B, Kx, X, Y=Y))

    def _matmul_lr(self, b: LRLinearMap) -> LRLinearMap:
        if self.B is None:
            return LRLinearMap(b.T(self.A.T).T, None)
        r = self.B @ b.A if b.B is None else self.B @ b.A @ b.B
        return LRLinearMap(self.A, r)

    def diag(self) -> np.ndarray:
        if self.B is None:
            return np.diag(self.A)
        return np.einsum('ij,ji->i', self.A, self.B)

    def T_(self) -> LRLinearMap:
        if self.B is None:
            return LRLinearMap(self.A.T, None)
        return LRLinearMap(self.B.T, self.A.T)

    T = property(T_)


def get_nystrom_L(z_nystrom_fn, Z1, Kz, nu, Z2=None, cg=False, jitter=1e-8):
    """
    return the evaluation on Z2 of the Nystrom predictor fitted on (Z1m, Z1, f(Z1)=?), which is
        ? |-> K(Z2, Z1m) @ (nu Kmm + Kmn @ Knm)^{-1} @ K(Z1m, Z1) @ ?
    """
    if Z2 is None:
        Z2 = Z1
    Z1m = z_nystrom_fn(Z1)
    Kmn = Kz(Z1m, Z1)
    B = (cg_solve if cg else np.linalg.solve)(
        nu * Kz(Z1m, Z1m) + Kmn @ Kmn.T + jitter * np.eye(Kmn.shape[0]),
        Kmn)
    assert not np.isnan(B.sum())
    return LRLinearMap(Kz(Z2, Z1m), B)


class KIVPredictor(object):

    """
    Implements kernelized [Quasi-Bayesian] dual IV.
    """

    def __init__(self, Z, X, Y, Kz, Kx, lam, nu, z_nystrom=None, cg=False, jitter=1e-8):
        n = Z.shape[0]
        vars(self).update(locals())  # save data and args
        if z_nystrom is None:
            Kxx = Kx(X, X)
            Kzz = Kz(Z, Z)
            I = np.eye(n)
            L = np.linalg.solve(Kzz + nu*I, Kzz)
            self.eff_prec = LRLinearMap(np.linalg.solve(lam*I + L@Kxx, L), None)
            self.L = LRLinearMap(L, None)
        else:
            nys_z = z_nystrom(Z)
            Kmn, Kmm = Kz(nys_z, Z), Kz(nys_z, nys_z); Knm = Kmn.T
            Iz = np.eye(Kmm.shape[0])
            Km_tilde = nu * Kmm + Kmn @ Knm + jitter * Iz
            # Let A^T A = L_tilde = Knm Kmtilde^{-1} Kmn
            A0 = np.linalg.cholesky(Km_tilde) # A0 A0.T = Km_tilde; A0^-T A0^-1 = Km_tilde^{-1}
            assert not np.isnan(A0.sum())
            A = jax.scipy.linalg.solve_triangular(A0, Kmn, lower=True)
            """
            Now eff_prec 
              = (lambda I + A^T A K_x)^{-1} A^T A 
              = lam^{-1} (A^T A - A^T (lam I + A Kx A^T)^{-1} A K_x A^T A)
              = lam^{-1} A^T tmp A
            """
            # evals, evecs = np.linalg.eigh(A @ Kx(X, X) @ A.T)
            evals, evecs = np.linalg.eigh(_matmul_with_Kxy(A, Kx, X) @ A.T)
            tmp = evecs @ np.diag(lam / (evals + lam)) @ evecs.T
            assert not np.isnan(tmp.sum())
            self.eff_prec = LRLinearMap(1/lam*A.T, tmp @ A)
            self.L = get_nystrom_L(z_nystrom, Z, Kz, nu, jitter=jitter)
        # 
        self.mean_base = self.eff_prec(Y)

    def to_onp(self):
        for k, v in vars(self).items():
            if isinstance(v, np.ndarray):
                setattr(self, k, onp.asarray(v))
                if k not in ['Z', 'X', 'Y', 'lam', 'nu']:
                    del v  # np arrays have a default ref count of 2
        self.eff_prec.to_onp()
        self.L.to_onp()

    def to_jnp(self):
        for k, v in vars(self).items():
            if isinstance(v, onp.ndarray):
                setattr(self, k, np.asarray(v))
        self.eff_prec.to_jnp()
        self.L.to_jnp()

    def _get_cov_subtacted_part(self, x_test) -> LRLinearMap:
        assert self.z_nystrom is not None
        return self.eff_prec._matmul_Kxy(self.Kx, self.X, Y=x_test).T._matmul_Kxy(
            self.Kx, self.X, Y=x_test).T

    def __call__(self, x_test, full_cov=False):
        """
        return: (pred_mean, pred_cov) if full_cov, else (pred_mean, pred_cov_diag)
        NOTE: pred_mean has shape [N, 1], but pred_cov_diag will have shape [N]
        """
        if self.z_nystrom is None:
            Ktx = self.Kx(x_test, self.X)
            mean = Ktx @ self.mean_base
            if not full_cov:
                cov_ret = self.Kx.kdiag(x_test) - np.diag(Ktx @ self.eff_prec(Ktx.T))
            else:
                cov_ret = self.Kx(x_test, x_test) - Ktx @ self.eff_prec(Ktx.T)
            return mean, cov_ret
        #
        mean_T = _matmul_with_Kxy(self.mean_base.T, self.Kx, self.X, Y=x_test)
        csp = self._get_cov_subtacted_part(x_test)
        if full_cov:
            return mean_T.T, (csp, 'handle it manually')
        return mean_T.T, self.Kx.kdiag(x_test) - csp.diag()
 
    def log_qlh(self):
        """
        return log \int Pi(df) exp(-(f(X)-Y) lam L^{-1} (f(X)-Y)). See _computation.tex.
        """
        quad_term = (self.Y * self.eff_prec(self.Y)).sum()
        logdet = np.log(self.L._matmul_Kxy(self.Kx, self.X).diag() + self.lam).sum() -\
                self.n * np.log(self.lam)
        return -1/2 * (logdet + quad_term)

    
kiv = KIVPredictor


def kiv_stage1_criteria(zx_tuples, nu, Kz, Kx, z_nystrom=None, cache_matrices=None, cg=False,
                        jitter=1e-8):
    (Z1, X1), (Z2, X2) = zx_tuples
    if z_nystrom is None:
        if cache_matrices is None:
            cache_matrices = (Kz(Z1, Z1), Kz(Z1, Z2), Kx(X2, X1), Kx(X1, X1))
        Kz11, Kz12, Kx21, Kx11 = cache_matrices
        L = np.linalg.solve(Kz11 + nu*np.eye(Z1.shape[0]), Kz12)
        return Kx.kdiag(X2).mean() + np.diag(-2 * Kx21 @ L + L.T @ Kx11 @ L).mean()
    else:
        L_T = get_nystrom_L(z_nystrom, Z1, Kz, nu, Z2, cg=cg, jitter=jitter)
        ret = Kx.kdiag(X2).mean() +\
            -2 * (L_T._matmul_Kxy(Kx, X1, X2).diag()).mean() + \
            L_T._matmul_Kxy(Kx, X1)._matmul_lr(L_T.T).diag().mean()
        return ret


def kiv_hps_selection(
    dtrain, dheldout, Kz, Kx, nu_space, lam_space=None, s2_criterion='orig',
    return_all_stats=False, z_nystrom=None, jitter=1e-8):
    r"""
    Hyperparameter selection for KIV. Selection of nu follows Algorithm 2 in the KIV paper.
    For lambda, we implement
    - 'orig': the original statistics in the KIV paper, n^{-1} \sum_i (y_i-\hat f(x_i))^2.
    - 'proj': what we used in the experiments, n^{-1} \sum_i (y_i - \hat E_n(\hat f)(z_i))^2
              this is closer to the kernelized dual IV objective
    """
    s1_stats = onp.array(
        [kiv_stage1_criteria(
            (dtrain[:2], dheldout[:2]), nu, Kz, Kx, z_nystrom=z_nystrom, jitter=jitter)
         for nu in nu_space])
    nu = nu_space[onp.nanargmin(s1_stats)]
    
    if lam_space is None:
        return ((nu, s1_stats) if return_all_stats else nu)
    
    (Z1, X1, Y1), (Z2, X2, Y2) = dtrain, dheldout
    
    if z_nystrom is None:
        K1, K2, K12 = Kz(Z1, Z1), Kz(Z2, Z2), Kz(Z1, Z2)
        L2 = np.linalg.solve(K2 + nu*np.eye(K2.shape[0]), K2).T
        L2 = LRLinearMap(L2, None)
    else:
        L2 = get_nystrom_L(z_nystrom, Z2, Kz, nu, jitter=jitter)

    proj_stats, orig_stats = [], []
    for lam in lam_space:
        pred_fn = kiv(Z1, X1, Y1, Kz, Kx, lam, nu, z_nystrom=z_nystrom, jitter=jitter)
        r"""
        proj: computes
            (\hat E \hat f)(z2) = Sz2 (Cz^{-1} Czx) mu \approx (empirical ver.) =: pf_cexp
        Note that we can use either dtrain or dheldout to estimate the condexp operator,
        as this is validation.  We use dheldout which seems closer to the typical sample 
        splitting regime.  Then we need to calculate L2 S_{x2} mu = L2 @ pred_fn(X2) where
        L2 = S_{z2} \invEmpCz2 n2^{-1}S_{z2} = K_{z2}(K_{z2}+nu I)^{-1}. 
        """
        pfx2_mean, pfx2_cov = pred_fn(X2, full_cov=True)
        pf_cexp = L2(pfx2_mean)
        y_cexp = L2(Y2)
        proj_stats.append(((pf_cexp-Y2)**2).mean())
        """ orig, from aux/KIV2_loss.m of the KIV codebase """
        orig_stats.append(((pfx2_mean-Y2)**2).mean())
        # work around some GPU memory issue
        pred_fn.to_onp()

    proj_stats, orig_stats = map(onp.array, (proj_stats, orig_stats))
    s2_stats = locals()[s2_criterion+'_stats']
    if onp.any(onp.isnan(s2_stats)):
        print('warning: nan encountered in kiv_hps_selection, crit=',
              s2_criterion, lam_space, s2_stats)
    lam = lam_space[onp.nanargmin(s2_stats)]
    
    if return_all_stats:
        return nu, lam, (proj_stats, orig_stats)
    else:
        return nu, lam


def cg_solve(A, b):
    return jax.scipy.sparse.linalg.cg(A, b)[0]


def krr(x, y, k, lam, cg=False, nystrom=None, jitter=1e-8):
    if nystrom is None:
        kx = k(x, x)
        solve = np.linalg.solve if not cg else cg_solve
        mean_base = solve(kx + lam*np.eye(kx.shape[0]), y)
        def predict(xtest):
            return k(xtest, x) @ mean_base
    else:
        x_nys = nystrom(x)
        L = get_nystrom_L(lambda _: x_nys, x, k, lam, cg=cg, jitter=jitter)
        def predict(xtest):
            return k(xtest, x_nys) @ (L.B @ y)
    return predict


class BootstrapPredictor(object):

    def __init__(self, pred_fns):
        self.pred_fns = pred_fns

    def __call__(self, inp, with_qb=False):
        preds = [p(inp) for p in self.pred_fns]
        pred_mean, pred_var = map(lambda a: onp.array(list(a)), zip(*preds))
        pmean = pred_mean.mean(0)
        pvar = (pred_mean.std(0)**2).squeeze()
        if with_qb:
            pvar += pred_var.mean(0)
        return pmean, pvar

    def to_jnp(self):
        [p.to_jnp() for p in self.pred_fns]

    def to_onp(self):
        [p.to_onp() for p in self.pred_fns]


def bootstrap(
    Ztr, Xtr, Ytr, Kz, Kx, lam, nu, ratio=0.95, n_repeats=20, z_nystrom=None, jitter=1e-8, rng=None):
    if rng is None:
        rng = onp.random.RandomState(23)
    pfns = []
    for _ in range(n_repeats):
        if ratio < 1:  # w/o replacement
            Dtrain, _ = data_split(Ztr, Xtr, Ytr, split_ratio=ratio, rng=rng)
        else:  # w/ replacement
            N = Ztr.shape[0]
            idcs = np.asarray(rng.randint(low=0, high=N, size=(N,)))
            Dtrain = (Ztr[idcs], Xtr[idcs], Ytr[idcs])
        pred_fn = kiv(Dtrain[0], Dtrain[1], Dtrain[2], Kz, Kx, lam, nu, z_nystrom=z_nystrom, jitter=jitter)
        pfns.append(pred_fn)
    return BootstrapPredictor(pfns)


if __name__ == '__main__':
    def sse(a, b): return ((a-b)**2).sum()
    rng = onp.random.RandomState(233)
    X = rng.normal(size=(1000, 3))
    sp = X.shape[0] // 2
    A = rng.normal(size=(500, 100)); A = A @ A.T
    kern = MaternKernel(5, x_train=X)
    ak0 = A @ kern(X[:sp], X[sp:])
    ak1 = _matmul_with_Kxy(A, kern, X[:sp], X[sp:], blk_size=599)
    ak2 = _matmul_with_Kxy(A, kern, X[:sp], X[sp:], blk_size=59)
    ak3 = _matmul_with_Kxy(A, kern, X[:sp], X[sp:], blk_size=250)
    print(sse(ak1, ak0), sse(ak2, ak0), sse(ak3, ak0))

    A = rng.normal(size=(500, 100))
    B = np.concatenate([A[:, :59], rng.normal(size=(500, 41))], 1).T
    C = A @ B
    Ct = LRLinearMap(A, B)
    print(sse(Ct.diag(), np.diag(C)))

    # the following will have large errors when using FP32
    B = np.concatenate([B, B], 1)
    C = A @ B
    Ct = LRLinearMap(A, B)
    R = Ct._matmul_Kxy(kern, X, X)
    print(((C @ kern(X, X) - R.A @ R.B)**2).sum())
    RR = R.T._matmul_lr(Ct)
    print(sse(RR.A @ RR.B, R.B.T @ R.A.T @ C))
