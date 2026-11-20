import numpy as np
from scipy.special import expit

D = 10
MU = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
C = np.array([
    [1., .8, 0, 0, 0, 0, 0, 0, 0, 0],
    [.8, 1., 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1., .3, .6, 0, 0, 0, 0, 0],
    [0, 0, .3, 1., .7, 0, 0, 0, 0, 0],
    [0, 0, .6, .7, 1., 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1., .5, 0, 0, 0],
    [0, 0, 0, 0, 0, .5, 1., 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1., -.4, 0],
    [0, 0, 0, 0, 0, 0, 0, -.4, 1., 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1.],
])
STD = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
SIGMA = np.diag(STD) @ C @ np.diag(STD)
THETA = np.array([.8, -.4, 1., 0., .6, -.5, .4, -.35, .2, -.18])
if not np.array_equal(SIGMA, SIGMA.T):
    raise ValueError("Asymmetric population covariance")
np.linalg.cholesky(SIGMA)
for constant in (MU, STD, C, SIGMA, THETA):
    constant.setflags(write=False)


def sample_x(n, rng):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator")
    return rng.multivariate_normal(MU, SIGMA, size=n)


def sample_y(x, rng):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != D or not np.isfinite(x).all():
        raise ValueError("Expected finite N by 10 covariates")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a NumPy Generator")
    return rng.binomial(1, expit(x @ THETA))


def design_parameters():
    return dict(distribution='single_gaussian', design='exploratory_scaled_gaussian', D=D,
                MU=MU.tolist(), STD=STD.tolist(), C=C.tolist(), SIGMA=SIGMA.tolist(),
                theta_true=THETA.tolist(), regression_intercept=False,
                response='Bernoulli(expit(X @ THETA))', response_coordinates='raw_X')
