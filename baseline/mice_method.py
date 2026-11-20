from untils.metrics import classification_metrics

from untils.time import timed_method, phase, in_phase, measure
import time
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from untils.inference import estimate_fields
from untils.inference import fit_vb_mc, predict_proba_mc


@timed_method
def run_mice_baseline(X_obs_train, y_train, X_obs_test, y_test,
                      train_mask_missing, test_mask_missing, prior_var, vb_cfg,
                      mice_cfg, device, theta_mc_samples=50,
                      regression_transform=None, raw_observed=None):
    started = time.perf_counter()
    masks = [np.asarray(train_mask_missing, bool), in_phase("test", lambda: np.asarray(test_mask_missing, bool))]
    inputs = [np.asarray(X_obs_train), in_phase("test", lambda: np.asarray(X_obs_test))]
    if masks[0].all(0).any():
        raise ValueError("MICE requires observations in every training column")
    nan_inputs = [in_phase("test" if i else "train", lambda: np.where(m, np.nan, x)) for i,(x,m) in enumerate(zip(inputs,masks))]
    completions = [[], []]
    iterations = []
    for j in range(int(mice_cfg.num_imputations)):
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=int(mice_cfg.max_iter),
                                  sample_posterior=True, random_state=int(mice_cfg.seed)+97*(j+1),
                                  imputation_order="ascending", initial_strategy="mean")
        train = imputer.fit_transform(nan_inputs[0])
        iterations.append(int(imputer.n_iter_))
        with phase("test"):
            imputer.random_state_ = np.random.RandomState((int(mice_cfg.seed)+97*(j+1)) ^ 0x5A5A5A5A)
            test = imputer.transform(nan_inputs[1])
        for i, x in enumerate([train,test]):
            with phase("test" if i == 1 else "train"):
                if x.shape != inputs[i].shape or not np.isfinite(x).all():
                    raise ValueError("Invalid MICE completion")
                x[~masks[i]] = inputs[i][~masks[i]]
                if regression_transform is not None:
                    x = regression_transform.unstandardize(x, raw_observed[i], masks[i])
                completions[i].append(x)
    train, test = [in_phase("test" if i else "train", lambda: np.stack(x)) for i,x in enumerate(completions)]
    imputation_sec = time.perf_counter()-started
    with measure(device) as measured:
        mu,cov,elbo = fit_vb_mc(train, y_train, prior_var, vb_cfg, device)
    inference_sec = measured['seconds']
    start = time.perf_counter()
    with phase("test"):
        p = predict_proba_mc(test, mu,cov,theta_mc_samples,
                             int(getattr(vb_cfg,"prediction_seed",vb_cfg.seed)))
    prediction_sec = time.perf_counter()-start
    means = [in_phase("test" if i else "train", lambda: x.mean(0)) for i,x in enumerate([train,test])]
    for i in range(2):
        with phase("test" if i == 1 else "train"):
            original = inputs[i] if raw_observed is None else raw_observed[i]
            means[i][~masks[i]] = original[~masks[i]]
    with phase("evaluation"):
        return dict(in_phase("evaluation", lambda: classification_metrics(y_test,p)), **estimate_fields(mu,cov,True),
                    method="mice_mc_vb", mu=mu,cov=cov, X_train_imp=means[0],X_test_imp=means[1],
                    X_train_samples=train,X_test_samples=test, num_imputations=len(train),
                    mice_iterations=iterations,
                    gvb_fits=1, regression_input_shape=list(train.shape),
                    imputation_sec=imputation_sec,inference_sec=inference_sec,prediction_sec=prediction_sec,
                    elbo_last=elbo,supports_oos=True)
