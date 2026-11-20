from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse, contextlib, csv, hashlib, importlib.metadata, json, os
import threading, time, traceback, warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any
from untils.data_process import validate_observed
from untils.metrics import classification_metrics
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
DATA = ROOT/'Data/support2'
OUTPUT = ROOT/'results/support2/support2_10.csv'
PROTOCOL_VERSION = 'support2_dvai_train_test_metrics_elbo_v3'
SEEDS = tuple(range(2027,2037))
DATA_SHA256 = 'f2f9bbfab6d2966b31e1858bf56fca61565301bb16f718b90f6ee0029bc3e6c1'
SOURCE_URL = 'https://hbiostat.org/data/repo/support2csv.zip'
NUMERIC15 = ['age','meanbp','wblc','hrt','resp','temp','pafi','alb','bili','crea','sod','ph','glucose','bun','urine']
COUNTS = ['num.co','hday']
RAW_FEATURES = NUMERIC15 + COUNTS + ['sex','diabetes','dementia','dzgroup','ca']
DZ = ['ARF/MOSF w/Sepsis','COPD','CHF','Cirrhosis','Coma','Colon Cancer','Lung Cancer','MOSF w/Malig']
DZ_COLUMNS = ['dzgroup_COPD','dzgroup_CHF','dzgroup_Cirrhosis','dzgroup_Coma','dzgroup_Colon_Cancer','dzgroup_Lung_Cancer','dzgroup_MOSF_w_Malig']
FEATURES = NUMERIC15 + ['num_co','hday','sex_male','diabetes','dementia'] + DZ_COLUMNS + ['ca_yes','ca_metastatic']
INDICATORS = ['sex_male','diabetes','dementia'] + DZ_COLUMNS + ['ca_yes','ca_metastatic']
TARGET = 'hospdead'
FROZEN = {'configuration': {'diffusion': {'hidden': 1024, 'lr': 0.0001, 'batch_size': 4096, 'epochs': 10000, 'sigma_min': 0.002, 'sigma_max': 80.0, 'edm_P_mean': -1.2, 'edm_P_std': 1.2, 'sigma_data': 0.5, 'sample_steps': 50, 'L': 6, 'langevin_sigma': 0.4, 'langevin_step_scale': 0.02, 'data_scale': 2.0, 'train_seed': None}, 'dvai': {'K': 10, 'J': 20, 'posterior_tol': 0.1, 'min_iter': 3, 'patience': 2, 'final_sample_clamp_val': None}, 'gvb': {'lr': 0.002, 'max_iter': 10000, 'num_mc': 20, 'check_every': 1, 'grad_clip': 0.0, 'window': 50, 'seed': 42, 'track_history': True, 'patience': 500, 'min_improve': 0.0, 'restore_best': False, 'jitter': 1e-06, 'beta1': 0.9, 'beta2': 0.9, 'step_adaptive': 500, 'init_scale': 0.1, 'denominator_floor': 1e-12, 'min_abs_diag': 1e-12, 'eigen_diagnostic_every': 0, 'final_elbo_samples': 2000, 'diagnostic_seed': 920260}}, 'evaluation': {'theta_mc_samples': 50, 'acc_threshold': 0.5}, 'prior': {'prior_var_used': 5.0}, 'stop': {'distance': 'DB/p', 'comparison': 'strict_less_than', 'posterior_pair': 'adjacent_iterations_same_run', 'covariance': 'full', 'p': 'regression_parameter_dimension_including_intercept', 'tolerance': 0.1, 'min_iter': 3, 'patience': 2, 'K_max': 10}}
SOURCE_HASHES = {'configs.py': '114a3aa26d1bb6f25c552978955cbe66cf3738ed528a4aebeb86ef49349a4618', 'models/diffusion_model.py': '694639ab38d9de711b69c9725c7d6b8f8d4ab0be5d9f92accb17db95ced544d3', 'models/diffusion_imputer.py': '30a13d775a5a498016b7a68db91eedb207289f90e3c060a893b41c2116b36b4f', 'models/diffusion_utils.py': '32cc0f7a396904e01a4aef8a5a7cd9532fbb95f52627b1813d311da7441f6098', 'untils/cholesky_gvb.py': 'bffc7439a19937cf3e439f414920b43daf76fa77c270108163d578dd5e37ca12', 'untils/logistic_regression.py': '70e5e7e1c9b52aa7d85b961b49f9064177f18096fb367d5eae3585c5cba83c91', 'untils/inference.py': 'b335fd80f86d2ee0ec73e447afd894e5bcc2602c041d6a8eac52eb7b01738bfb', 'untils/metrics.py': 'e6e171c5cf7ddb223bf7c59184a77ff1fb5f71af6329d14431693d7a69d5a3c1', 'untils/data_process.py': '0d4217590c350cc5179f23257f4c6af0b69a21ed44775e9d38c82e547830f0d6', 'untils/seed_device.py': '957c217d2139ad12bbd85022087f4551e9ee4a6d075bdf055945052fc4bd8d59', 'untils/time.py': 'd848fb7bb66cc2596edd2ba9d39cbdce20589e9412d551b96e3860489cdf014b', 'baseline/mean_method.py': '66dbb0d0da8a8767c91d1946c5cc3f8952677c4e671a498aec49b05b711b4782'}
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def json_value(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)

def add_intercept(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim not in (2,3) or x.shape[-1] != 29 or not np.isfinite(x).all():
        raise ValueError('Expected finite 29-column covariate array')
    return np.concatenate([np.ones((*x.shape[:-1],1),dtype=np.float32),x],axis=-1)

def _observed_mean_std(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    validate_observed(X_train,np.isnan(X_train),training=True,nan_missing=True)
    mean = np.nanmean(X_train, axis=0)
    std = np.nanstd(X_train, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return mean.astype(float), std.astype(float)

def _add_intercept(X: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    X_out = np.concatenate([np.ones((n, 1), dtype=np.float32), X.astype(np.float32)], axis=1)
    mask_out = np.concatenate([np.zeros((n, 1), dtype=bool), mask.astype(bool)], axis=1)
    return X_out, mask_out

def preprocess_split(
    X_raw: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_names: list[str],
    add_intercept: bool = True,
    data_scale: float = 1.0,
    context: str = '',
) -> dict[str, Any]:
    X_raw = np.asarray(X_raw, dtype=float)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if data_scale <= 0:
        raise ValueError("data_scale must be positive.")

    X_train_full = X_raw[train_idx]
    for split,x in [('train',X_train_full),('test',X_raw[test_idx])]:
        validate_observed(x,np.isnan(x),training=split=='train',feature_names=feature_names,
                          context=f'{context} split={split}',nan_missing=True)
    train_var = np.nanvar(X_train_full, axis=0)
    train_zero_var = ~np.isfinite(train_var) | (train_var <= 1e-12)
    keep_cols = ~train_zero_var
    if not np.any(keep_cols):
        raise ValueError(
            "No covariate columns remain after dropping train-only invalid columns."
        )

    kept_feature_names = [
        name for name, keep in zip(feature_names, keep_cols) if bool(keep)
    ]
    dropped_cols = [
        name for name, keep in zip(feature_names, keep_cols) if not bool(keep)
    ]
    X_kept = X_raw[:, keep_cols]

    X_train_raw = X_kept[train_idx]
    X_test_raw = X_kept[test_idx]
    mean, std = _observed_mean_std(X_train_raw)

    def transform(X_part: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isnan(X_part)
        X_std = ((X_part - mean) / std) / float(data_scale)
        X_zero = np.where(mask, 0.0, X_std).astype(np.float32)
        return X_zero, mask.astype(bool)

    X_train_obs_zero, train_mask_cov = transform(X_train_raw)
    X_test_obs_zero, test_mask_cov = transform(X_test_raw)

    train_mask = train_mask_cov
    test_mask = test_mask_cov
    model_feature_names = list(kept_feature_names)
    if add_intercept:
        X_train_obs_zero, train_mask = _add_intercept(
            X_train_obs_zero, train_mask_cov
        )
        X_test_obs_zero, test_mask = _add_intercept(
            X_test_obs_zero, test_mask_cov
        )
        model_feature_names = ["intercept"] + model_feature_names

    metadata = {
        "D_raw_covariate": int(X_raw.shape[1]),
        "D_covariate": int(len(kept_feature_names)),
        "D_model": int(X_train_obs_zero.shape[1]),
        "add_intercept": bool(add_intercept),
        "intercept_mask_all_false": bool(
            (not add_intercept) or not train_mask[:, 0].any()
        ),
        "data_scale": float(data_scale),
        "kept_feature_names": kept_feature_names,
        "model_feature_names": model_feature_names,
        "dropped_cols": dropped_cols,
        "observed_counts_train": (~train_mask_cov).sum(0).tolist(),
        "observed_counts_test": (~test_mask_cov).sum(0).tolist(),
        "test_all_missing_columns": [name for name,mask in zip(kept_feature_names,test_mask_cov.T) if mask.all()],
        "dropped_cols_train_zero_variance": [
            name for name, bad in zip(feature_names, train_zero_var) if bool(bad)
        ],
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    return {
        "X_train_obs_zero": X_train_obs_zero,
        "X_test_obs_zero": X_test_obs_zero,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "train_mask_covariates": train_mask_cov,
        "test_mask_covariates": test_mask_cov,
        "y_train": y[train_idx],
        "y_test": y[test_idx],
        "metadata": metadata,
    }

def prepare_inputs(d,train,test,data_scale):
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    x=d[FEATURES].to_numpy(dtype=np.float64);y=d[TARGET].to_numpy(dtype=np.int64)
    for label,index in [('train',train),('test',test)]:
        v=x[index];counts=np.isfinite(v).sum(0)
        if (counts==0).any():raise ValueError(f'{label} all-missing columns: {np.array(FEATURES)[counts==0].tolist()}')
        variance=np.nanvar(v,axis=0)
        if (~np.isfinite(variance)|(variance<=1e-12)).any():
            raise ValueError(f'{label} zero/invalid variance; stop without dropping columns')
    prep=preprocess_split(x,y,train,test,FEATURES,add_intercept=False,data_scale=data_scale,context='support2 preparation')
    meta=prep['metadata']
    if meta['dropped_cols'] or meta['D_covariate']!=29:raise ValueError('Public preprocessing unexpectedly changed dimensions')
    assert np.array_equal(np.asarray(meta['mean']),np.nanmean(x[train],axis=0))
    assert np.array_equal(np.asarray(meta['std']),np.nanstd(x[train],axis=0))
    for label,index in [('train',train),('test',test)]:
        u=prep[f'X_{label}_obs_zero'];mask=prep[f'{label}_mask']
        assert u.shape==mask.shape==(len(index),29)
        assert mask.dtype==bool and np.array_equal(mask,np.isnan(x[index]))
        assert np.isfinite(u).all() and not mask[:,[FEATURES.index(c) for c in INDICATORS]].any()
        assert np.array_equal(u[mask],np.zeros(mask.sum(),dtype=np.float32))
        design=add_intercept(u)
        assert design.shape==(len(index),30) and (design[:,0]==1).all()
        back=u.astype(float)*np.asarray(meta['std'])*data_scale+np.asarray(meta['mean'])
        bound=8*np.finfo(np.float32).eps*(np.abs(x[index])+np.abs(np.asarray(meta['mean']))+np.asarray(meta['std'])*data_scale+1.)
        errors=np.abs(back-x[index])
        assert np.all(errors[~mask]<=bound[~mask])
        meta[label+'_float32_inverse_maxabs_by_column']=np.nanmax(errors,axis=0).tolist()
        assert np.allclose(back[:,[FEATURES.index(c) for c in INDICATORS]],x[index][:,[FEATURES.index(c) for c in INDICATORS]],atol=2e-6)
    prep['raw_train']=x[train];prep['raw_test']=x[test]
    return prep

def validate_completed(samples,observed,mask,raw,meta):
    a=np.asarray(samples)
    if a.ndim not in (2,3) or a.shape[-2:]!=observed.shape or observed.shape!=mask.shape or mask.shape[1]!=29:
        raise ValueError('Completed-data dimension mismatch')
    if not np.isfinite(a).all():raise FloatingPointError('Nonfinite completed values')
    actual=a[:,~mask] if a.ndim==3 else a[~mask]
    expected=observed[~mask]
    tolerance=32*np.finfo(np.float32).eps*max(1.,float(np.max(np.abs(expected))))
    error=float(np.max(np.abs(actual-expected))) if expected.size else 0.
    if error>tolerance:raise AssertionError(f'Observed values changed: {error}, tolerance={tolerance}')
    ind=[FEATURES.index(c) for c in INDICATORS]
    if mask[:,ind].any():raise AssertionError('Unexpected missing categorical values')
    inverse=a[...,ind].astype(float)*np.asarray(meta['std'])[ind]*meta['data_scale']+np.asarray(meta['mean'])[ind]
    if not np.allclose(inverse,raw[:,ind],rtol=2e-6,atol=2e-6):raise AssertionError('Original categories changed')
    return {'max_observed_error':error,'float32_tolerance':tolerance,'categories_preserved':True}

def validate_posterior(mu,cov):
    mu,cov=np.asarray(mu),np.asarray(cov)
    if mu.shape!=(30,) or cov.shape!=(30,30):raise ValueError('Posterior dimension mismatch')
    if not np.isfinite(mu).all() or not np.isfinite(cov).all():raise FloatingPointError('Nonfinite posterior')
    if not np.allclose(cov,cov.T,rtol=1e-5,atol=1e-7):raise ValueError('Asymmetric covariance')
    np.linalg.cholesky(cov.astype(float))

def digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode()+a.tobytes()).hexdigest()

def design_checked(x):
    design = add_intercept(x)
    expected = np.asarray(x, dtype=np.float32)
    conversion_error = np.abs(expected.astype(np.float64)-np.asarray(x,dtype=np.float64))
    bound = np.finfo(np.float32).eps*np.abs(np.asarray(x,dtype=np.float64))+np.finfo(np.float32).smallest_subnormal
    assert np.all(conversion_error <= bound)
    assert design.shape[-1] == 30 and np.array_equal(design[..., 1:], expected)
    assert np.all(design[..., 0] == 1)
    return design

def factor_check(vb, mu, cov):
    C = vb._build_L().detach().cpu().numpy().copy()
    validate_posterior(mu, cov)
    assert np.isfinite(C).all() and np.array_equal(C, np.tril(C))
    assert np.allclose(C@C.T, cov, rtol=1e-5, atol=1e-7)
    eigen = np.linalg.eigvalsh(np.asarray(cov, dtype=float))
    assert eigen.min() > 0 and np.min(np.abs(np.diag(C))) > 1e-12
    return C, {'posterior_mean_shape': list(mu.shape), 'posterior_covariance_shape': list(cov.shape),
        'min_covariance_eigenvalue': float(eigen.min()), 'max_covariance_eigenvalue': float(eigen.max()),
        'covariance_condition_number': float(eigen.max()/eigen.min()),
        'min_abs_factor_diagonal': float(np.min(np.abs(np.diag(C)))),
        'C_C_transpose_matches_covariance': True, 'strict_Cholesky_without_jitter': True}

def initial_check(vb):
    mu = vb.mu.detach().cpu().numpy().copy()
    C = vb._build_L().detach().cpu().numpy().copy()
    assert np.array_equal(mu, np.zeros(30, np.float32))
    assert np.array_equal(C, np.eye(30, dtype=np.float32)*np.float32(.1))
    return {'mean_sha256': digest(mu), 'factor_sha256': digest(C), 'mean_all_zero': True,
            'factor_exact_0_1_identity': True, 'posterior_warm_start': False}

def finite_gvb(out,budget):
    for key in ['mu','cov','elbo_history','elbo_smooth_history','min_abs_diag_history',
                'learning_rate_history','final_elbo','final_elbo_mcse','best_monitor_value','last_monitor_value']:
        if out.get(key) is not None:assert np.isfinite(np.asarray(out[key],dtype=float)).all(),key
    assert 1<=out['actual_steps']<=budget

def input_hashes(prep):
    assert 'y_test' not in prep
    return {key:digest(prep[key]) for key in ['X_train_obs_zero','X_test_obs_zero','train_mask','test_mask']}

def fit_gvb(design,y,vc,device,prior_var,mc,run):
    import torch
    from untils.logistic_regression import BayesianLogisticRegressionMC,BayesianLogisticRegression
    from untils.cholesky_gvb import CholeskyGaussianVB
    from untils.time import phase
    klass=BayesianLogisticRegressionMC if mc else BayesianLogisticRegression
    task=klass(torch.as_tensor(design,device=device,dtype=torch.float32),
               torch.as_tensor(y,device=device,dtype=torch.float32),prior_var=prior_var)
    assert not task.use_full_prior and task.prior_var==5.
    vb=CholeskyGaussianVB(30,vc,device);run.last_vb=vb
    with phase('evaluation'):
        initialization=initial_check(vb)
    out=vb.fit(task)
    with phase('evaluation'):
        finite_gvb(out,vc.max_iter)
        C,numerical=factor_check(vb,out['mu'],out['cov'])
    return out,C,numerical,initialization

def environment():
    import torch
    return {'executable':sys.executable,'python':sys.version,'torch_file':torch.__file__,'torch_C_file':torch._C.__file__,
        'torch_version':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
        'device_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'torch_distribution_versions':[{'version':d.version,'metadata_path':str(d._path)} for d in importlib.metadata.distributions() if d.metadata.get('Name','').lower()=='torch'],
        'dependency_versions':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','matplotlib']},
        'environment_repaired':False}

def execute_dvai(prep,settings,run):
    from models.diffusion_imputer import ConditionalDiffusionImputer
    from untils.inference import StopTracker,checked_predict
    from untils.metrics import posterior_components
    from untils.seed_device import derive_seed,set_global_seed
    from untils.time import phase
    diff,dvai,gvb,cfg=settings
    assert len(prep['y_train'])==6373 and prep['X_test_obs_zero'].shape==(2732,29)
    before=input_hashes(prep)
    seed=lambda purpose,*keys:derive_seed(run.algorithm_seed,'formal',run.rep,purpose,'support2',*keys)
    set_global_seed(seed('setup'))
    obs,mask=prep['X_train_obs_zero'],prep['train_mask']
    completed=obs.copy();previous=None;records=run.iterations
    tracker=StopTracker(dvai.posterior_tol,dvai.patience,dvai.min_iter)
    for k in range(1,dvai.K+1):
        seeds={purpose:seed(purpose,k) for purpose in ['network_init','training','sampling_train','sampling_test','langevin_train','langevin_test','vb','prediction']}
        dc,vc=deepcopy(diff),deepcopy(gvb)
        dc.init_seed,dc.train_seed,vc.seed=seeds['network_init'],seeds['training'],seeds['vb']
        assert dc.epochs==10000 and dc.hidden==1024 and vc.max_iter==10000 and dvai.K==10 and dvai.J==20
        input_digest=digest(completed)
        if records:assert input_digest==records[-1]['completion_mean_sha256']
        run.last_completion=completed
        with run.stage('DVAI','diffusion_training',iteration=k):
            imputer=ConditionalDiffusionImputer(29,dc,run.device);run.last_network=imputer
            init_digest=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in imputer.model.state_dict().values())).hexdigest()
            trained=imputer.fit(completed)
            assert 1<=trained['actual_epochs']<=10000
            assert np.isfinite([trained['best_loss'],trained['last_loss']]).all()
        with run.stage('DVAI','training_conditional_sampling',iteration=k):
            set_global_seed(seeds['sampling_train'])
            draws=np.ascontiguousarray(imputer.sample(np.asfortranarray(obs),np.asfortranarray(mask),dvai.J,
                                                     langevin_seed=seeds['langevin_train']))
            if dvai.final_sample_clamp_val is not None:
                draws[:,mask]=np.clip(draws[:,mask],-dvai.final_sample_clamp_val,dvai.final_sample_clamp_val)
            with phase('evaluation'):
                anchor=validate_completed(draws,obs,mask,prep['raw_train'],prep['metadata'])
                assert np.array_equal(draws[:,~mask],np.broadcast_to(obs[~mask],draws[:,~mask].shape))
            completed=draws.mean(0).astype(np.float32);completed[~mask]=obs[~mask]
            design=design_checked(draws);run.last_completion=completed
        with run.stage('DVAI','GVB',iteration=k):
            out,C,numerical,initialization=fit_gvb(design,prep['y_train'],vc,run.device,cfg.realdata_run_config.prior_var_used,True,run)
        with run.stage('DVAI','outer_stopping_and_checks','diagnostic',k):
            mu,cov=out['mu'],out['cov']
            distance=posterior_components(*previous,mu,cov)['db_per_dim'] if previous is not None else None
            previous=mu.copy(),cov.copy();stop=tracker.update(k,distance)
            assert before==input_hashes(prep)
        record={'iteration':k,'seeds':seeds,'actual_configs':{'diffusion':vars(dc),'gvb':vars(vc)},
            'diffusion':trained,'gvb':{key:value for key,value in out.items() if key not in ['mu','cov'] and (not key.endswith('_history') or key=='elbo_history')},
            'gvb_initialization':initialization,'numerical_checks':numerical,'train_anchor_before':anchor,'train_anchor_after':anchor,
            'training_input_sha256':input_digest,'completion_mean_sha256':digest(completed),
            'previous_completion_used':None if not records else input_digest==records[-1]['completion_mean_sha256'],
            'fresh_network_initial_sha256':init_digest,'posterior_DB_per_30':distance,'streak':tracker.streak,
            'algorithm_stop_met':bool(stop),'termination':'algorithm_stop_criterion' if stop else ('outer_budget_limit' if k==dvai.K else 'continue')}
        run.event({'event':'outer_iteration_completed','iteration':k,'distance':distance,'streak':tracker.streak,'stop':bool(stop)})
        print(f'DVAI k={k} DB/p={distance} streak={tracker.streak} stop={stop}',flush=True)
        records.append(record)
        if stop or k==dvai.K:break
        run.last_network=None;run.last_vb=None
        del imputer,draws,design,out
    with run.stage('DVAI','test_conditional_sampling','test',k):
        set_global_seed(seeds['sampling_test'])
        te_obs,te_mask=prep['X_test_obs_zero'],prep['test_mask']
        test_draws=np.ascontiguousarray(imputer.sample(np.asfortranarray(te_obs),np.asfortranarray(te_mask),dvai.J,
                                                      langevin_seed=seeds['langevin_test']))
        if dvai.final_sample_clamp_val is not None:
            test_draws[:,te_mask]=np.clip(test_draws[:,te_mask],-dvai.final_sample_clamp_val,dvai.final_sample_clamp_val)
        with phase('evaluation'):
            after_anchor=validate_completed(test_draws,te_obs,te_mask,prep['raw_test'],prep['metadata'])
            assert np.array_equal(test_draws[:,~te_mask],np.broadcast_to(te_obs[~te_mask],test_draws[:,~te_mask].shape))
    with run.stage('DVAI','test_prediction','test',k):
        probabilities=checked_predict(design_checked(test_draws),mu,cov,seeds['prediction'],cfg.eval_config.theta_mc_samples)
        assert probabilities.shape==(2732,) and np.isfinite(probabilities).all() and ((probabilities>=0)&(probabilities<=1)).all()
        assert before==input_hashes(prep)
    with run.stage('DVAI','training_evaluation','diagnostic',k):
        train_p=checked_predict(design,mu,cov,seed('train_prediction',k),cfg.eval_config.theta_mc_samples)
        train_scores=classification_metrics(prep['y_train'],train_p,auc_only=True)
    result={'mean':mu,'covariance':cov,'factor':C,'prediction':probabilities,'train_prediction':train_p,'iterations':records,
        'train_AUC':train_scores['AUC'],
        'train_AUC_reason':None if train_scores['AUC'] is not None else 'single_class_train_labels',
        'actual_iterations':k,'stop_reason':'algorithm_stop_criterion' if stop else 'outer_budget_limit',
        'algorithm_stop_met':bool(stop),'statistical_convergence_claimed':False,
        'test_anchor_before':after_anchor,'test_anchor_after':after_anchor,'original_masks_unchanged':True,**run.totals('DVAI')}
    run.last_network=None;run.last_vb=None
    return result


def json_text(value):
    return json.dumps(value, ensure_ascii=False, default=json_value, allow_nan=False,
                      separators=(',', ':'), sort_keys=True)


def resolve_configs():
    import configs
    diff, dvai, gvb = (deepcopy(value) for value in (
        configs.diff_config, configs.dvai_config, configs.dvai_cholesky_gvb_config))
    actual = {'configuration': {'diffusion': asdict(diff), 'dvai': asdict(dvai),
                               'gvb': asdict(gvb)},
              'evaluation': asdict(configs.eval_config),
              'prior': asdict(configs.realdata_run_config),
              'stop': configs.dvai_stop_protocol()}
    assert actual == FROZEN, 'Live configuration differs from validated pilot'
    for name, expected in SOURCE_HASHES.items():
        assert sha256(ROOT/name) == expected, f'Previously verified source changed: {name}'
    return diff, dvai, gvb, configs


def load_data():
    assert sha256(DATA/'support2.csv') == DATA_SHA256, 'Analysis data changed'
    d = pd.read_csv(DATA/'support2.csv', float_precision='round_trip',
                   keep_default_na=False, na_values=['NaN'])
    assert list(d.columns) == FEATURES+[TARGET] and d.shape == (9105, 30)
    x, y = d[FEATURES].to_numpy(float), d[TARGET].to_numpy()
    assert not np.isinf(x).any() and np.isin(y, [0, 1]).all() and y.sum() == 2360
    assert np.isnan(x).sum() == 24580
    indicators = x[:, [FEATURES.index(c) for c in INDICATORS]]
    assert np.isin(indicators, [0, 1]).all()
    assert (d[DZ_COLUMNS].sum(axis=1) <= 1).all()
    assert (d[['ca_yes', 'ca_metastatic']].sum(axis=1) <= 1).all()
    ids = np.asarray([f'SUPPORT2_{i+1}' for i in range(len(d))])
    return d, ids


def make_split(d, seed):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(np.arange(9105), train_size=.7,
                                 stratify=d[TARGET].to_numpy(), random_state=seed)
    assert len(train) == 6373 and len(test) == 2732
    assert len(np.unique(np.concatenate([train, test]))) == 9105
    split_hash = hashlib.sha256(np.asarray([len(train), len(test)], dtype='<i8').tobytes()
                               + np.asarray(train, dtype='<i8').tobytes()
                               + np.asarray(test, dtype='<i8').tobytes()).hexdigest()
    return train, test, split_hash


def configuration(env):
    return {**FROZEN, 'protocol_version': PROTOCOL_VERSION, 'environment': env, 'features': FEATURES,
            'raw_features': RAW_FEATURES, 'source_url': SOURCE_URL,
            'source_hashes': SOURCE_HASHES, 'runner_sha256': sha256(__file__),
            'patient_id_rule': 'SUPPORT2_{one-based unchanged original row}',
            'anomalies': 'All original observations retained; flags are not verified errors',
            'category_reference': {'sex': 'female', 'dzgroup': DZ[0], 'ca': 'no'},
            'split_seeds': SEEDS, 'algorithm_master_seeds': SEEDS,
            'seed_protocol': "derive_seed(algorithm_seed,'formal',rep,purpose,'support2',k)",
            'initialization': {'network_each_round': 'fresh', 'GVB_each_round': 'mu=0,C=0.1I',
                               'handoff': 'completion mean only', 'prior_update': False},
            'sampler_refine_steps': 1, 'diffusion_patience': 500,
            'result_metrics': {'train': ['train_AUC'], 'test': list(METRIC_NAMES)},
            'timing': 'CUDA-synchronized project measure/phase; train+test; excludes preprocessing, outer stopping, postfit diagnostics, evaluation and file I/O',
            'determinism_call_order': 'setup=True; sampling=True',
            'statistical_convergence_claimed': False}


class Run:
    def __init__(self, rep, seed, device):
        self.rep, self.algorithm_seed, self.device = rep, seed, device
        self.records, self.iterations = [], []
        self.active = None
        self.last_network = self.last_vb = self.last_completion = None
        self.stop_event = threading.Event()
        self.main_thread = threading.get_ident()

    def event(self, value):
        print(f'rep={self.rep} '+json_text(value), flush=True)

    @contextlib.contextmanager
    def stage(self, method, name, kind='train', iteration=None):
        from untils.time import measure, phase
        self.active = {'method': method, 'stage': name, 'iteration': iteration}
        self.event({**self.active, 'event': 'start'})
        measured, status = {}, 'failed'
        started = time.perf_counter()
        try:
            with measure(self.device) as measured:
                with phase('evaluation' if kind == 'diagnostic' else kind):
                    yield
            status = 'passed'
        finally:
            record = {**self.active, 'status': status,
                      'wall_seconds': time.perf_counter()-started,
                      'train_seconds': measured.get('train_sec', 0.),
                      'test_seconds': measured.get('test_sec', 0.),
                      'excluded_diagnostic_seconds': measured.get('excluded_sec', 0.)}
            self.records.append(record)
            self.event({**record, 'event': 'finish'})

    def totals(self, method='DVAI'):
        train = sum(r['train_seconds'] for r in self.records if r['method'] == method)
        test = sum(r['test_seconds'] for r in self.records if r['method'] == method)
        return {'train_time_sec': train, 'test_time_sec': test, 'total_time_sec': train+test}

    def heartbeat(self):
        while not self.stop_event.wait(30):
            frame = sys._current_frames().get(self.main_thread)
            progress = {}
            while frame is not None:
                module, name = frame.f_globals.get('__name__'), frame.f_code.co_name
                if (module, name) in [('models.diffusion_imputer', 'train_diffusion'),
                                     ('untils.cholesky_gvb', 'fit')]:
                    for key in ['epoch', 'step', 'best', 'bad', 'wait']:
                        value = frame.f_locals.get(key)
                        if isinstance(value, (int, float)) and np.isfinite(value):
                            progress[key] = value
                frame = frame.f_back
            self.event({'event': 'progress', 'active': self.active, **progress})

    def start_monitor(self):
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()

    def close_monitor(self):
        self.stop_event.set()
        self.thread.join(timeout=5)


METRIC_NAMES = ['AUC', 'ACC', 'Brier']
FIELDS = ['record_type', 'rep', 'status', 'split_seed', 'algorithm_seed',
          'n_train', 'n_test', 'n_events_train', 'n_events_test',
          'train_AUC', 'train_AUC_reason', 'AUC', 'AUC_reason', 'ACC', 'Brier',
          'outer_iterations', 'stop_reason', 'train_time_sec', 'test_time_sec', 'total_time_sec',
          'data_sha256', 'split_hash', 'configuration_json', 'preprocessing_json',
          'posterior_mean_json', 'posterior_covariance_json', 'parameter_names_json',
          'iteration_history_json', 'warnings_json', 'error_message', 'checks_json',
          'split', 'patient_id', 'y_true', 'p_pred', 'n_missing']


def read_rows(path=OUTPUT):
    csv.field_size_limit(32*1024*1024)
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FIELDS, 'Unexpected result CSV schema'
        return list(reader)


def atomic_save(rows, expected_hash):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    current = sha256(OUTPUT) if OUTPUT.exists() else None
    assert current == expected_hash, 'Result changed externally; refusing overwrite'
    temp = OUTPUT.with_name(OUTPUT.name+'.tmp')
    try:
        with temp.open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        check = read_rows(temp)
        assert len(check) == len(rows)
        assert (sha256(OUTPUT) if OUTPUT.exists() else None) == expected_hash
        os.replace(temp, OUTPUT)
    finally:
        if temp.exists():
            temp.unlink()
    return sha256(OUTPUT)


def validate_saved(rows, d, ids, config_text, *, only_rep=None):
    runs, predictions = {}, {}
    for row in rows:
        rep = int(row['rep'])
        assert 1 <= rep <= 10
        if row['record_type'] == 'run':
            assert rep not in runs, 'Duplicate run'
            runs[rep] = row
        else:
            assert row['record_type'] == 'prediction'
            assert row.get('split') in ('train', 'test'), 'Old predictions lack train/test split labels; use a separate result file for a new run'
            predictions.setdefault(rep, {}).setdefault(row['split'], []).append(row)
    assert set(predictions) <= set(runs)
    assert sorted(runs) == list(range(1, len(runs)+1)), 'Nonsequential repetitions'
    for rep, row in runs.items():
        if only_rep is not None and rep != only_rep:
            continue
        seed = SEEDS[rep-1]
        train, test, split_hash = make_split(d, seed)
        assert int(row['split_seed']) == int(row['algorithm_seed']) == seed
        assert row['data_sha256'] == DATA_SHA256 and row['split_hash'] == split_hash
        saved_config, current_config = json.loads(row['configuration_json']), json.loads(config_text)
        assert saved_config.get('protocol_version') == PROTOCOL_VERSION, 'Old results lack separate train/test metrics or predictions; move them to your backup before a new run'
        assert saved_config == current_config, 'Configuration/environment changed'
        assert json.loads(row['parameter_names_json']) == ['intercept']+FEATURES
        if row['status'] != 'completed':
            assert rep == max(runs) and rep not in predictions
            assert row['status'] in ['running', 'interrupted', 'failed']
            continue
        assert int(row['n_train']) == 6373 and int(row['n_test']) == 2732
        assert np.isfinite(float(row['train_AUC'])) and 0 <= float(row['train_AUC']) <= 1
        times = [float(row[key]) for key in ('train_time_sec','test_time_sec','total_time_sec')]
        assert np.isfinite(times).all() and min(times) >= 0 and np.isclose(times[2],times[0]+times[1])
        history = json.loads(row['iteration_history_json'])
        assert history['elbo_fit_split'] == 'train'
        iterations = history['iterations']
        assert [r['iteration'] for r in iterations] == list(range(1,int(row['outer_iterations'])+1))
        for iteration in iterations:
            trace = iteration['gvb']['elbo_history']
            assert len(trace) == iteration['gvb']['actual_steps'] > 0 and np.isfinite(trace).all()
        saved_splits = predictions.get(rep, {})
        assert set(saved_splits) == {'train', 'test'}
        for split, index in [('train', train), ('test', test)]:
            saved = saved_splits[split]
            assert len(saved) == len({r['patient_id'] for r in saved}) == len(index)
            assert [r['patient_id'] for r in saved] == ids[index].tolist()
            y = np.asarray([int(r['y_true']) for r in saved])
            assert np.array_equal(y, d.iloc[index][TARGET].to_numpy())
            assert int(row[f'n_events_{split}']) == int(y.sum())
            missing = d.iloc[index][FEATURES].isna().sum(axis=1).to_numpy()
            assert np.array_equal(missing, [int(r['n_missing']) for r in saved])
            p = np.asarray([float(r['p_pred']) for r in saved])
            metrics = classification_metrics(y, p, auc_only=split == 'train')
            for name in ('AUC',) if split == 'train' else METRIC_NAMES:
                column = 'train_AUC' if split == 'train' else name
                value = metrics[name]
                if value is None:
                    assert row[column] == ''
                    assert row[column+'_reason'] == f'single_class_{split}_labels'
                else:
                    assert np.isclose(float(row[column]), value, rtol=1e-10, atol=1e-12), column
        validate_posterior(np.asarray(json.loads(row['posterior_mean_json'])),
                           np.asarray(json.loads(row['posterior_covariance_json'])))
        prep = prepare_inputs(d, train, test, 2.)
        assert json.loads(row['preprocessing_json']) == preprocessing_record(prep)
        assert json.loads(row['checks_json'])['all_required_checks_passed']
    return runs


def preprocessing_record(prep):
    return {**prep['metadata'], 'formula': 'U=(X-mu_train)/(2*s_train)',
            'ddof': 0, 'statistics_source': 'training observed values only',
            'diffusion_dimension': 29, 'regression_dimension': 30}


def check_only():
    resolve_configs()
    d, ids = load_data()
    for rep, seed in enumerate(SEEDS, 1):
        train, test, split_hash = make_split(d, seed)
        prep = prepare_inputs(d, train, test, 2.)
        assert prep['metadata']['kept_feature_names'] == FEATURES
        print(json_text({'rep': rep, 'seed': seed, 'train': len(train), 'test': len(test),
                         'split_hash': split_hash, 'checks': 'passed', 'training': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reps', type=int, choices=range(1,11), default=10)
    args = parser.parse_args()
    if args.check:
        check_only()
        return 0
    import torch
    from untils.seed_device import configure_single_thread
    configure_single_thread()
    settings = resolve_configs()
    d, ids = load_data()
    env = environment()
    assert env['cuda_available'], 'Validated CUDA device unavailable; no silent CPU fallback'
    config_text = json_text(configuration(env))
    rows, expected_hash, old_runs = [], None, {}
    if OUTPUT.exists():
        assert args.resume, 'Existing result: use --resume for validation, never silently overwrite'
        rows, expected_hash = read_rows(), sha256(OUTPUT)
        old_runs = validate_saved(rows, d, ids, config_text)
        assert not any(r['status'] == 'failed' for r in old_runs.values()), 'Substantive failure saved; automatic retry prohibited'
    for rep, seed in enumerate(SEEDS[:args.reps], 1):
        if rep in old_runs and old_runs[rep]['status'] == 'completed':
            print(f'rep={rep}: verified completed result; skipping', flush=True)
            continue
        train, test, split_hash = make_split(d, seed)
        row = dict(record_type='run', rep=rep, status='running', split_seed=seed,
                   algorithm_seed=seed, n_train=6373, n_test=2732,
                   n_events_train=int(d.iloc[train][TARGET].sum()),
                   n_events_test=int(d.iloc[test][TARGET].sum()), data_sha256=DATA_SHA256,
                   split_hash=split_hash, configuration_json=config_text,
                   parameter_names_json=json_text(['intercept']+FEATURES),
                   warnings_json='[]', iteration_history_json='[]', checks_json='{}')
        rows = [r for r in rows if int(r['rep']) != rep]+[row]
        expected_hash = atomic_save(rows, expected_hash)
        run = Run(rep, seed, torch.device('cuda'))
        notes = []
        if rep in old_runs:
            notes.append('Explicit resume of interrupted repetition; no successful prediction was reused')
        if len(env['torch_distribution_versions']) > 1:
            notes.append('Pre-existing duplicate torch metadata; imported '+str(env['torch_version'])+'; environment unchanged')
        run.start_monitor()
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                prep = prepare_inputs(d, train, test, settings[0].data_scale)
                prep.pop('y_test')
                row['preprocessing_json'] = json_text(preprocessing_record(prep))
                result = execute_dvai(prep, (*[deepcopy(s) for s in settings[:3]], settings[3]), run)
                y = d.iloc[test][TARGET].to_numpy(dtype=int)
                metrics = classification_metrics(y, result['prediction'])
                checks = {'all_required_checks_passed': True, 'original_masks_unchanged': result['original_masks_unchanged'],
                          'test_anchor_before': result['test_anchor_before'], 'test_anchor_after': result['test_anchor_after'],
                          'posterior': result['iterations'][-1]['numerical_checks'],
                          'column_order_and_dimensions': True, 'no_degenerate_columns': True,
                          'intercept_only_at_regression_boundary': True, 'train_only_preprocessing': True,
                          'original_patient_outcome_alignment': True, 'finite_probabilities_in_0_1': True,
                          'no_test_labels_in_fit_or_stopping': True,
                          'statistical_convergence_claimed': False}
                row.update({name: metrics[name] for name in (*METRIC_NAMES, 'AUC_reason')},
                           status='completed', train_AUC=result['train_AUC'],
                           train_AUC_reason=result['train_AUC_reason'], outer_iterations=result['actual_iterations'],
                           stop_reason=result['stop_reason'], train_time_sec=result['train_time_sec'],
                           test_time_sec=result['test_time_sec'], total_time_sec=result['total_time_sec'],
                           posterior_mean_json=json_text(result['mean']),
                           posterior_covariance_json=json_text(result['covariance']), checks_json=json_text(checks))
                predictions = [dict(record_type='prediction', rep=rep, split=split, patient_id=ids[i], y_true=int(labels[j]),
                                    p_pred=float(probabilities[j]), n_missing=int(prep[f'{split}_mask'][j].sum()))
                               for split, index, labels, probabilities in [
                                   ('train', train, prep['y_train'], result['train_prediction']),
                                   ('test', test, y, result['prediction'])]
                               for j, i in enumerate(index)]
                assert len(predictions) == len({r['patient_id'] for r in predictions}) == len(d)
                resolve_configs()
                assert sha256(DATA/'support2.csv') == DATA_SHA256
                rows.extend(predictions)
        except BaseException as exc:
            row.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                       error_message=traceback.format_exc(), stop_reason='execution_error',
                       outer_iterations=len(run.iterations), checks_json=json_text({'failed_stage': run.active}))
            row.update(run.totals())
            traceback.print_exc()
            rows = [r for r in rows if not (int(r['rep']) == rep and r['record_type'] == 'prediction')]
        finally:
            run.close_monitor()
            notes.extend(str(w.message) for w in caught)
            row['warnings_json'] = json_text(notes)
            row['iteration_history_json'] = json_text({'elbo_fit_split': 'train', 'iterations': run.iterations, 'stages': run.records})
            expected_hash = atomic_save(rows, expected_hash)
        print(json_text({k: row.get(k) for k in ['rep', 'status', 'train_AUC', *METRIC_NAMES,
                                               'outer_iterations', 'total_time_sec']}), flush=True)
        if row['status'] != 'completed':
            return 1
        validate_saved(read_rows(), d, ids, config_text, only_rep=rep)
        run.last_network = run.last_vb = run.last_completion = None
        del result, prep, run
    verified = validate_saved(read_rows(), d, ids, config_text)
    assert all(verified[rep]['status'] == 'completed' for rep in range(1,args.reps+1))
    assert len(rows) == (len(d)+1)*len(verified)
    print(f'Completed {args.reps} requested DVAI repetitions; {len(rows)} records in {OUTPUT}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

