import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression

# from benchmarks.poisson_regression import PoissonRegressionLineSearch


class Model:

    def __init__(self, M_t, weighting, **kwargs):
        self.M_t = M_t
        # weighting per municipality should not influence regularization
        self.weighting = weighting / np.sum(weighting) * M_t.shape[0]

    def fit_predict(self, m, m_obs_ixs, m_unobs_ixs):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError


class Averaging(Model):

    def fit_predict(self, m, m_obs_ixs, m_unobs_ixs):
        pred = m[m_obs_ixs].mean(axis=0)
        pred_col = np.ones_like(m) * pred
        pred_col[m_obs_ixs] = m[m_obs_ixs]
        return pred_col

    @property
    def name(self):
        return 'Averaging'


class WeightedAveraging(Model):

    def fit_predict(self, m, m_obs_ixs, m_unobs_ixs):
        wts = self.weighting[m_obs_ixs]
        N = wts.sum()
        if len(m.shape) > 1:
            raise ValueError('Multi-dim not available.')
        pred = (m[m_obs_ixs] * wts).sum() / N
        pred_col = np.ones_like(m) * pred
        pred_col[m_obs_ixs] = m[m_obs_ixs]
        return pred_col

    @property
    def name(self):
        return 'Weighted Averaging'


class MatrixFactorisation(Model):
    # defaults to Referenda hyperparameters (Etter et al.)

    def __init__(self, M_t, weighting, n_iter=20, n_dim=25,
                 lam_U=0.0316, lam_V=31.6):
        super().__init__(M_t, weighting)
        self.n_iter_cache = n_iter
        if len(M_t.shape) > 2:
            raise ValueError('Tensor not factorisable.')
        # initiatialize like netflix prize papers
        U = np.random.rand(len(M_t), n_dim+1) / (n_dim+1)
        U[:, -1] = 1.0
        V = np.random.rand(len(M_t[0])+1, n_dim+1) / (n_dim+1)
        self.U, self.V = U, V
        self.lam_U = lam_U
        self.lam_V = lam_V
        self.n_dim = n_dim

    @property
    def weighted(self):
        return False

    def fit_predict(self, m, m_obs_ixs, m_unobs_ixs):
        for i in range(self.n_iter_cache):
            observed = np.zeros_like(m, dtype=bool)
            observed[m_obs_ixs] = True
            self.update_U(m, observed)
            self.update_V(m, observed)
        return self.U @ self.V[-1, :]

    def update_U(self, m, observed):
        # (1) update representations of fully observed regions
        M = np.concatenate([self.M_t[observed], m[observed].reshape(-1, 1)], 1)
        U = self.U[observed, :-1]
        V, b = self.V[:, :-1], self.V[:, -1]
        ones = np.ones(len(U))
        B = ((M - np.outer(ones, b)) @ V).T
        A = V.T @ V + self.lam_U * np.eye(self.n_dim)
        self.U[observed, :-1] = np.linalg.solve(A, B).T

        # (2) update representations of other regions
        M = self.M_t[~observed]
        U = self.U[~observed, :-1]
        # only take prior vote representations to update
        V, b = self.V[:-1, :-1], self.V[:-1, -1]
        ones = np.ones(len(U))
        B = ((M - np.outer(ones, b)) @ V).T
        A = V.T @ V + self.lam_U * np.eye(self.n_dim)
        self.U[~observed, :-1] = np.linalg.solve(A, B).T

    def update_V(self, m, observed):
        # (1) update all prior vote representations (fully observed ones)
        V = self.V[:-1]
        B = (self.M_t.T @ self.U).T
        I = np.eye(self.n_dim+1)
        # don't regularize bias
        I[-1, -1] = 0
        A = self.U.T @ self.U + self.lam_V * I
        self.V[:-1] = np.linalg.solve(A, B).T

        # (2) update the new vote representation
        U = self.U[observed]
        V = self.V[-1:]
        B = (m[observed].reshape(-1, 1).T @ U).T
        A = U.T @ U + self.lam_V * I
        self.V[-1:] = np.linalg.solve(A, B).T

    @property
    def name(self):
        name =  'Matrix Factorization' if not self.weighted \
                else 'Weighted Matrix Factorization'
        name += (' (dim=' + str(self.n_dim) + ',lam_V=' + str(self.lam_V) +
                 ',lam_U=' + str(self.lam_U) + ')')
        return name


class SubSVD(Model):

    def __init__(self, M_t, weighting,
                 n_dim=10, add_bias=True, l2_reg=1e-5, keep_svals=True):
        if len(M_t.shape) > 2:
            raise ValueError('Tensor not factorisable.')
        super().__init__(M_t, weighting)
        U, s, _ = np.linalg.svd(M_t)
        self.n_dim = n_dim
        self.U = U[:, :n_dim] * s[None, :n_dim] if keep_svals else U[:, :n_dim]
        self.l2_reg = l2_reg
        self.add_bias = add_bias

    def fit_predict(self, m, m_obs_ixs, m_unobs_ixs):
        Uo, mo = self.U[m_obs_ixs], m[m_obs_ixs]
        if self.l2_reg is not None and self.l2_reg != 0:
            ridge = Ridge(alpha=self.l2_reg, fit_intercept=self.add_bias)
            ridge.fit(Uo, mo)
            return ridge.predict(self.U)
        else:
            x, _, _, _ = np.linalg.lstsq(Uo, mo, 1e-9)
            return self.U @ x

    @property
    def name(self):
        return ('SubSVD' + ' (dim=' + str(self.n_dim) + ',l2=' +
                str(self.l2_reg) + ')')


class WeightedSubSVD(SubSVD):

    def fit_predict(self, m, m_obs_ixs, m_unobs_ixs):
        Uo, mo, wo = self.U[m_obs_ixs], m[m_obs_ixs], self.weighting[m_obs_ixs]
        if self.l2_reg is not None and self.l2_reg != 0:
            ridge = Ridge(alpha=self.l2_reg, fit_intercept=self.add_bias)
            ridge.fit(Uo, mo, sample_weight=wo)
            return ridge.predict(self.U)
        else:
            wo_sqrt = np.sqrt(wo)
            x, _, _, _ = np.linalg.lstsq(
                Uo * wo_sqrt.reshape(-1, 1), mo * wo_sqrt, 1e-9)
            return self.U @ x

    @property
    def name(self):
        return ('Weighted SubSVD' + ' (dim=' + str(self.n_dim) + ',l2=' +
                str(self.l2_reg) + ')')


class LogisticSubSVD(SubSVD):

    @staticmethod
    def transform_problem(Uo, mo, wo=None):
        # Repeat dataset for label y==1 and y==0 and use the probabilities
        # as weights. This is equal to cross-entropy logistic regression.
        n = Uo.shape[0]
        wo = np.ones(n) if wo is None else wo / wo.sum() * n
        wo = np.tile(wo, 2)
        y = np.zeros(2*n)
        y[:n] = 1
        wts = np.tile(mo, 2)
        wts[n:] = 1 - wts[n:]
        wts = wts * wo
        X = np.tile(Uo, (2, 1))
        return X, y, wts

    def fit_predict(self, m, m_obs_ixs, m_unobs_ixs):
        Uo, mo = self.U[m_obs_ixs], m[m_obs_ixs]
        C = 0 if self.l2_reg == 0 else 1 / self.l2_reg
        logreg = LogisticRegression(C=C, fit_intercept=self.add_bias,
                                    solver='liblinear', tol=1e-6, max_iter=500)
        X, y, wts = self.transform_problem(Uo, mo, None)
        logreg.fit(X, y, sample_weight=wts)
        return logreg.predict_proba(self.U)[:, 1]

    @property
    def name(self):
        return ('Logistic SubSVD' + ' (dim=' + str(self.n_dim) + ',l2=' +
                str(self.l2_reg) + ')')


class WeightedLogisticSubSVD(LogisticSubSVD):

    def fit_predict(self, m, m_obs_ixs, m_unobs_ixs):
        Uo, mo, wo = self.U[m_obs_ixs], m[m_obs_ixs], self.weighting[m_obs_ixs]
        C = 0 if self.l2_reg == 0 else 1 / self.l2_reg
        logreg = LogisticRegression(C=C, fit_intercept=self.add_bias,
                                    solver='liblinear', tol=1e-6, max_iter=500)
        X, y, wts = self.transform_problem(Uo, mo, wo)
        logreg.fit(X, y, sample_weight=wts)
        return logreg.predict_proba(self.U)[:, 1]

    @property
    def name(self):
        return ('Weighted Logistic SubSVD' + ' (dim=' + str(self.n_dim) + ',l2=' +
                str(self.l2_reg) + ')')


class TensorSubSVD(Model):
    """Folds the m by v by party tensor into m by (v*party) and then
    factorize"""

    def __init__(self, M_t, weighting, n_dim=10, add_bias=True, l2_reg=1e-5,
                 keep_svals=True):
        if len(M_t.shape) < 3:
            raise ValueError('Requires Tensor')
        super().__init__(M_t, weighting)
        M_t = M_t.reshape(M_t.shape[0], -1)
        U, s, _ = np.linalg.svd(M_t)
        self.U = U[:, :n_dim] * s[None, :n_dim] if keep_svals else U[:, :n_dim]
        self.l2_reg = l2_reg
        self.add_bias = add_bias
        self.n_dim = n_dim

    def fit_predict(self, m, m_obs_ixs, m_unobs_ixs):
        Uo, mo = self.U[m_obs_ixs], m[m_obs_ixs]
        if self.l2_reg is not None and self.l2_reg != 0:
            ridge = Ridge(alpha=self.l2_reg, fit_intercept=self.add_bias)
            ridge.fit(Uo, mo)
            return ridge.predict(self.U)
        else:
            # TODO: add bias here?
            x, _, _, _ = np.linalg.lstsq(Uo, mo, 1e-9)
            return self.U @ x

    @property
    def name(self):
        return ('SubSVD' + ' (dim=' + str(self.n_dim) + ',l2=' +
                str(self.l2_reg) + ')')


class LogisticTensorSubSVD(TensorSubSVD):

    def __init__(self, M_t, weighting, n_dim=10, add_bias=True, l2_reg=1e-5,
                 keep_svals=True):
        super().__init__(M_t, weighting, n_dim, add_bias, l2_reg, keep_svals)
        C = 0 if self.l2_reg == 0 else 1 / self.l2_reg
        self.model = LogisticRegression(C=C, fit_intercept=add_bias, tol=1e-6,
                                        solver='newton-cg', max_iter=5000,
                                        multi_class='multinomial', n_jobs=4,
                                        warm_start=True)

    @staticmethod
    def transform_problem(Uo, mo):
        n, k = mo.shape
        # classes 0*n, 1*n, ..., k*n
        y = np.arange(k).repeat(n)
        # weights are probabilities of respective class
        wts = mo.reshape(-1, order='F')
        # repeat data
        X = np.tile(Uo, (k, 1))
        return X, y, wts

    def fit_predict(self, m, m_obs_ixs, m_unobs_ixs):
        Uo, mo = self.U[m_obs_ixs], m[m_obs_ixs]
        X, y, wts = self.transform_problem(Uo, mo)
        self.model.fit(X, y, sample_weight=wts)
        return self.model.predict_proba(self.U)

    @property
    def name(self):
        return ('Logistic SubSVD' + ' (dim=' + str(self.n_dim) + ',l2=' +
                str(self.l2_reg) + ')')

