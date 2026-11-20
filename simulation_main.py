from __future__ import annotations

from untils.metrics import imputation_error_stats, posterior_components

import argparse
import hashlib
import inspect
import json
import time
import warnings
import uuid
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from untils.metrics import classification_metrics, METRIC_PROTOCOL, CRPS_PROTOCOL
from untils.time import TIMING_PROTOCOL, SENSITIVITY_TIMING_PROTOCOL

from untils.time import timed_method, phase, in_phase, measure
import configs as cfg
from generate_miss.fixed_design import D, SIGMA, THETA, sample_x, sample_y, design_parameters
from generate_miss.data_miss import mar_target_rate
from generate_miss.mask_mechanisms import fixed_missing_mask, summarize_missingness, missingness_parameters, validate_missing_rate
from models.diffusion_imputer import ConditionalDiffusionImputer
from untils.cholesky_gvb import CholeskyGaussianVB
from untils.result_io import (ResultCSV, result_directory, runtime_identity,
                              read_result_rows, completed_repetitions, sensitivity_path)
from untils.logistic_regression import BayesianLogisticRegression, BayesianLogisticRegressionMC
from untils.data_process import ObservedTransform, validate_observed
from untils.inference import StopTracker, checked_predict
from untils.seed_device import derive_seed, set_global_seed, configure_single_thread, collect_training

METHODS = ['mean', 'mice', 'missforest', 'miwae', 'diffputer', 'dvai']
PROTOCOL_VERSION = 'exploratory_scaled_gaussian_standardized_mnar_v2_evaluation_v4_train_auc_elbo_csv'
METRIC_SCALE = 'population_std'
PREDICTION_THETA_MC = cfg.eval_config.theta_mc_samples


def protocol_identity(effective, scenarios):
    identity = {k: v for k, v in effective.items()
                if k not in {'out_dir', 'rep_start', 'reps', 'retry_failed',
                             'methods', 'mechanisms', 'rates', 'scenarios', 'save_predictions'}}
    identity['file_scenarios'] = sorted(scenarios)
    return identity


def report_saved(store, mechanism, rate, rep, method, status):
    counts = completed_repetitions(store.rows, METHODS, mechanism, rate)
    print(f'[saved] {mechanism} {rate:.2f} | rep={rep} | method={method} | status={status}', flush=True)
    print(f'[completed] {mechanism} {rate:.2f} | ' +
          ' | '.join(f'{name}={count}' for name, count in counts.items()), flush=True)


def validate_terminal_record(row):
    if row.get('row_type') != 'final':
        return
    for key in ('train_impute_rmse', 'train_impute_mae',
                'test_impute_rmse', 'test_impute_mae', 'test_impute_crps', 'ACC', 'Brier',
                'train_time_sec', 'test_time_sec', 'total_time_sec'):
        if row.get(key) is None or not np.isfinite(float(row[key])):
            raise ValueError(f'Invalid final {key}: {row.get(key)}')
    if row.get('AUC') in (None, ''):
        if row.get('AUC_reason') != 'single_class_test_labels':
            raise ValueError('Missing AUC without a valid reason')
    elif not np.isfinite(float(row['AUC'])) or not 0 <= float(row['AUC']) <= 1:
        raise ValueError('Invalid AUC')
    if row.get('train_AUC') in (None, ''):
        if row.get('train_AUC_reason') != 'single_class_train_labels':
            raise ValueError('Missing training AUC without a valid reason')
    elif not np.isfinite(float(row['train_AUC'])) or not 0 <= float(row['train_AUC']) <= 1:
        raise ValueError('Invalid training AUC')
    if not all(0 <= float(row[k]) <= 1 for k in ('ACC', 'Brier')):
        raise ValueError('Invalid classification metric')
    train, test, total = (float(row[k]) for k in ('train_time_sec', 'test_time_sec', 'total_time_sec'))
    if min(train, test) < 0 or not np.isclose(total, train + test, rtol=1e-8, atol=1e-6):
        raise ValueError('Invalid train/test/total timing')
    if str(row.get('posterior_available')).lower() in {'true', '1'}:
        if row.get('overlap') is None or not np.isfinite(float(row['overlap'])) or not 0 <= float(row['overlap']) <= 1:
            raise ValueError('Invalid posterior Overlap')
    elif row.get('overlap') not in (None, ''):
        raise ValueError('Overlap requires a parameter posterior')


def encode(value):
    return json.dumps(value, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item(),
                      sort_keys=True, allow_nan=False, separators=(',', ':'))


from untils.result_io import source_version as code_version


def fit_vb(x, y, vb_cfg, device, diagnostics=None, *, save_elbo_history=False):
    klass = BayesianLogisticRegressionMC if x.ndim == 3 else BayesianLogisticRegression
    task = klass(torch.as_tensor(x, dtype=torch.float32, device=device),
                 torch.as_tensor(y, dtype=torch.float32, device=device), prior_var=5.)
    out = CholeskyGaussianVB(x.shape[-1], vb_cfg, device).fit(task)
    with phase("evaluation"):
        if diagnostics is not None:
            excluded = {'mu', 'cov'} if save_elbo_history else {'mu', 'cov', 'elbo_history'}
            diagnostics.update({k:v for k,v in out.items() if k not in excluded})
        posterior_components(out['mu'], out['cov'], out['mu'], out['cov'])['db_per_dim']
    return out['mu'], out['cov']


def predict(x, mean, cov, seed, count=50):
    return checked_predict(x, mean, cov, seed, count)


def add_training_evaluation(row, truth, fill, mask, samples, y, mean, cov, seed,
                            save_predictions=False):
    with phase('evaluation'):
        if 'train_impute_rmse' not in row:
            row.update(imputation_error_stats(truth, fill, mask, np.sqrt(np.diag(SIGMA)), 'train'))
        probabilities = predict(samples, mean, cov, seed, PREDICTION_THETA_MC)
        scores = classification_metrics(y, probabilities, auc_only=True)
        row.update(train_AUC=scores['AUC'],
                   train_AUC_reason=None if scores['AUC'] is not None else 'single_class_train_labels')
        if save_predictions:
            row.update(train_probabilities=encode(probabilities), train_labels=encode(y))


def add_estimate(row, mean, cov, reference, p, y, save_predictions=False, *, posterior_available=True):
    mean = np.asarray(mean)
    if posterior_available and cov is None:
        raise ValueError('Missing posterior covariance')
    row.update(posterior_available=posterior_available, theta_estimate=encode(mean),
               posterior_mean=encode(mean) if posterior_available else None,
               posterior_cov=encode(cov) if posterior_available else None,
               overlap=None, missingness_model="ignored")
    if posterior_available and reference is not None:
        row.update(posterior_components(mean, cov, *reference))
    if reference is None:
        row['overlap_reason'] = 'Full-VB reference failed or unavailable'
    if p is not None:
        row.update({k:v for k,v in classification_metrics(y,p).items() if k != 'p_test'})
        row['acc_threshold'] = .5
        if save_predictions:
            row.update(test_labels=encode(y), test_probabilities=encode(p))


def validate_candidate(candidate):
    if set(candidate) != {'name', 'J'} or candidate['J'] not in [5, 10, 20, 40]:
        raise ValueError('Candidate requires name and J in 5,10,20,40')


def sampling_fields(num_steps=50, diffusion=None):
    from models.diffusion_utils import dvai_sampling_metadata
    settings = deepcopy(cfg.diff_config if diffusion is None else diffusion)
    settings.sample_steps = num_steps
    return dvai_sampling_metadata(settings)


def repetition_data(args, rep):
    stream = lambda purpose: derive_seed(args.master_seed, args.stage, rep, purpose)
    seeds = {p:stream(p) for p in ['data', 'response', 'reference_vb', 'reference_prediction']}
    x = sample_x(args.n_train+args.n_test, np.random.default_rng(seeds['data']))
    y = sample_y(x, np.random.default_rng(seeds['response']))
    return x[:args.n_train], x[args.n_train:], y[:args.n_train], y[args.n_train:], seeds


def prepare_scenario(args, rep, mechanism, rate, xtr, xte, configs):
    mask_seeds = {s:derive_seed(args.master_seed,args.stage,rep,'mask',mechanism,s)
                  for s in ('train','test')}
    masks = [fixed_missing_mask(x,mechanism,rate,np.random.default_rng(mask_seeds[s]).random(x.shape))
             for s,x in zip(('train','test'),(xtr,xte))]
    context = f'dataset=simulation rep={rep} mechanism={mechanism} target_rate={rate}'
    for split,x,mask in zip(('train','test'),(xtr,xte),masks):
        validate_observed(x,mask,training=split=='train',context=f'{context} split={split}',
                          mechanism=mechanism,target_rate=rate)
    transform = ObservedTransform(xtr,masks[0],configs['diff_config'].data_scale,context=context)
    summaries = {s:summarize_missingness(m) for s,m in zip(('train','test'),masks)}
    metadata = dict(missingness_summary=encode(summaries),
                    missingness_parameters=encode(missingness_parameters(mechanism,rate)))
    for split,mask in zip(('train','test'),masks):
        metadata.update({f'{split}_actual_missing_rate': summaries[split]['overall_missing_rate'],
                         f'{split}_all_missing_rows': int(mask.all(1).sum()),
                         f'{split}_all_missing_columns': encode(np.flatnonzero(mask.all(0))+1)})
    return masks, transform, metadata, mask_seeds


def local_configs(args):
    configs = {name: deepcopy(getattr(cfg, name)) for name in [
        'diff_config', 'dvai_cholesky_gvb_config', 'mice_baseline_config',
        'missforest_baseline_config', 'miwae_baseline_config',
        'diffputer_baseline_config']}
    if args.stage == 'smoke':
        d = configs['diff_config']; d.hidden = 16; d.epochs = 2; d.sample_steps = 3
        v = configs['dvai_cholesky_gvb_config']; v.max_iter = 12; v.num_mc = 4
        configs['mice_baseline_config'].max_iter = 2
        configs['mice_baseline_config'].num_imputations = 2
        configs['missforest_baseline_config'].n_estimators = 5
        configs['missforest_baseline_config'].max_iter = 2
        m = configs['miwae_baseline_config']; m.epochs = 2; m.hidden_dim = 16; m.monitor_every = 1
        m.n_imputations_train = m.n_imputations_test = 5; m.num_importance_samples = 5
    if args.stage == "smoke":
        d = configs["diffputer_baseline_config"]
        d.K=2; d.hidden=16; d.epochs=2; d.sample_steps=3; d.refine_steps=2
    return configs


def data_digest(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        h.update(str((value.dtype.str, value.shape)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def full_vb_identity(xtr, ytr, seeds, configs, device):
    config = asdict(configs['dvai_cholesky_gvb_config'])
    config['seed'] = seeds['reference_vb']
    root = Path(__file__).resolve().parent
    source = hashlib.sha256(inspect.getsource(fit_vb).encode())
    for name in ('untils/cholesky_gvb.py', 'untils/logistic_regression.py',
                 'untils/seed_device.py'):
        source.update(name.encode())
        source.update((root / name).read_bytes())
    return dict(data=data_digest(xtr, ytr), coordinates='raw_X', intercept=False,
                prior_var=5., config=config, fit_protocol='full_vb_raw_X_v1',
                source_sha256=source.hexdigest(), dtype='float32', device=str(device),
                versions=runtime_identity(device),
                reference_vb_seed=seeds['reference_vb'])


def validate_full_vb(saved, identity):
    payload = encode(identity)
    if (saved.get('reference_identity') != payload or
            saved.get('reference_id') != hashlib.sha256(payload.encode()).hexdigest()):
        raise ValueError('Full-VB data/coordinates/prior/GVB/source/random-stream identity mismatch')
    mean, cov = np.asarray(json.loads(saved['reference_mean'])), np.asarray(json.loads(saved['reference_cov']))
    p = len(THETA)
    if mean.shape != (p,) or cov.shape != (p,p):
        raise ValueError('Invalid saved Full-VB dimensions')
    posterior_components(mean, cov, mean, cov)
    return mean, cov


def comparison_reference(xtr, ytr, seeds, configs, device):
    path = sensitivity_path('sigma', 'MAR')
    if not path.exists():
        return None
    rows = [r for r in read_result_rows(path) if r.get('trajectory') == 'reference/0']
    if not rows:
        return None
    if len(rows) != 1 or rows[0].get('status') != 'completed':
        raise ValueError('Ambiguous or unsuccessful historical Full-VB reference')
    saved = rows[0]
    protocol = json.loads(saved['effective_config'])
    if hashlib.sha256(encode(protocol).encode()).hexdigest() != saved['protocol_id']:
        raise ValueError('Historical reference protocol payload/hash mismatch')
    identity = full_vb_identity(xtr, ytr, seeds, configs, device)
    validate_full_vb(saved, identity)
    if saved['generation_parameters'] != encode(design_parameters()):
        raise ValueError('Historical data design differs')
    return saved


@timed_method
def run_baseline(method, xtr, xte, ytr, yte, masks, configs, seed, device, transform):
    from baseline.diffputer_method import run_diffputer_baseline
    from baseline.mean_method import run_mean_baseline
    from baseline.mice_method import run_mice_baseline
    from baseline.missforest_method import run_missforest_baseline
    from baseline.miwae_method import run_miwae_baseline
    functions = dict(diffputer=run_diffputer_baseline, mean=run_mean_baseline, mice=run_mice_baseline, missforest=run_missforest_baseline,
                     miwae=run_miwae_baseline)
    kwargs = dict(X_obs_train=xtr, X_obs_test=xte, y_train=ytr, y_test=yte,
                  train_mask_missing=masks[0], test_mask_missing=masks[1],
                  prior_var=5., vb_cfg=configs['dvai_cholesky_gvb_config'], device=device,
                  theta_mc_samples=50)
    kwargs.update(X_obs_train=transform.standardize(xtr, masks[0]),
                  X_obs_test=in_phase("test", lambda: transform.standardize(xte, masks[1])),
                  regression_transform=transform, raw_observed=(xtr, xte))
    for key in ['mice', 'missforest', 'miwae', 'diffputer']:
        kwargs[key+'_cfg'] = configs[key+'_baseline_config']
    set_global_seed(seed['training'])
    signature = inspect.signature(functions[method])
    out = functions[method](**{k: v for k, v in kwargs.items() if k in signature.parameters})
    for key, original, mask in zip(['X_train_imp','X_test_imp'],[xtr,xte],masks):
        with phase('test' if key == 'X_test_imp' else 'train'):
            out[key] = np.asarray(out[key],dtype=float)
            out[key][~mask] = original[~mask]
    return out


def run_trajectory(args, candidate, method, rep, mechanism, rate, xtr, xte, ytr, yte,
                   masks, reference, configs, metadata, rows, progress=None, transform=None):
    progress = {} if progress is None else progress
    progress.update(attempted_k=0, failure_stage="initialization")
    start = time.perf_counter()
    seed = lambda purpose, *keys: derive_seed(args.master_seed, args.stage, rep, purpose, mechanism, rate, method, *keys)
    base = dict(metadata, candidate=candidate['name'], method=method, mechanism=mechanism,
                target_missing_rate=rate, rep=rep, **{k: candidate.get(k) for k in ['J']},
                K_max=args.K_max,
                reference_mean=encode(reference[0]) if reference is not None else None,
                reference_cov=encode(reference[1]) if reference is not None else None,
                posterior_mean=None, posterior_cov=None, theta_estimate=None, overlap=None,
                convergence_distance=None, consecutive=0, first_stop_k=None, stop_reason=None,
                is_final=False, actual_k=0, attempted_k=0, failure_stage=None,
                status='running', warning=None, training_sec=None, imputation_sec=None,
                inference_sec=None, prediction_sec=None, diagnostic_sec=None,
                AUC=None, ACC=None, acc_threshold=.5, posterior_available=False)
    base.update(sampling_fields(configs['diff_config'].sample_steps, configs['diff_config']))
    if method not in {'dvai','diffputer'}:
        for key in ['J','K_max',*sampling_fields().keys()]: base[key] = None
    elif method == 'diffputer':
        base['J'] = configs['diffputer_baseline_config'].num_imputations
        base['K_max'] = configs['diffputer_baseline_config'].K
        base.update({key: None for key in sampling_fields()})
        base.update(fixed_resampling=configs['diffputer_baseline_config'].refine_steps, effective_resampling_counts=[configs['diffputer_baseline_config'].refine_steps]*(configs['diffputer_baseline_config'].sample_steps-1)+[1])
    base['reference_prior_var'] = 5.
    base['covariate_working_model'] = method
    base['missingness_model'] = 'ignored'
    rows.append(dict(base, row_type='initialization', k=0))
    if transform is None:
        raise ValueError('A shared observed-training transform is required')
    base.update(preprocess_center=encode(transform.center), preprocess_std=encode(transform.std),
                preprocess_degenerate_columns=encode(transform.degenerate))
    notes = ['Degenerate observed std replaced by 1: '+str(transform.degenerate)] if transform.degenerate else []
    ztr, zte = transform.standardize(xtr, masks[0]), transform.standardize(xte, masks[1])
    rawtr, rawte = transform.unstandardize(ztr, xtr, masks[0]), transform.unstandardize(zte, xte, masks[1])
    diagnostic_total = 0.
    def metrics(row, tr, te, test_samples=None, independent=False):
        nonlocal diagnostic_total
        t = time.perf_counter()
        if not getattr(args, 'compare_mcar_mar50', False):
            row.update(imputation_error_stats(xtr, tr, masks[0], np.sqrt(np.diag(SIGMA)), 'train'))
        row.update(imputation_error_stats(xte, te, masks[1], np.sqrt(np.diag(SIGMA)), 'test', samples=test_samples, independent=independent))
        row['diagnostic_sec'] = time.perf_counter()-t
        diagnostic_total += row['diagnostic_sec'] + row.get('extra_diagnostic_sec',0.)
        row['cumulative_sec'] = time.perf_counter()-start
        row['cumulative_diagnostic_sec'] = diagnostic_total
        row['algorithm_sec'] = row.get('total_time_sec')
        row['evaluation_timing_scope' if 'total_time_sec' in row else 'timing_scope'] = TIMING_PROTOCOL
        return row
    if method != 'dvai' and not getattr(args, 'compare_mcar_mar50', False):
        rows[0] = metrics(dict(base, row_type='initialization', k=0), rawtr, rawte)
    if method != 'dvai':
        progress.update(attempted_k=0 if method == 'diffputer' else 1, failure_stage='baseline_fit')
        seeds = {p: seed(p) for p in ['network_init','training','sampling','vb','prediction']}
        if method == 'miwae': seeds['sampling_test'] = seeds['sampling'] ^ 0xA5A5A5A5
        if method == 'mice':
            seeds['imputer_seeds'] = [seeds['training']+97*(i+1) for i in range(configs['mice_baseline_config'].num_imputations)]
        for c in configs.values():
            if hasattr(c, 'seed'): c.seed = seeds['training']
            if hasattr(c, 'random_state'): c.random_state = seeds['training']
        configs['dvai_cholesky_gvb_config'].seed = seeds['vb']
        for c in configs.values():
            c.sampling_seed = seeds['sampling']; c.init_seed = seeds['network_init']; c.prediction_seed = seeds['prediction']
        base['actual_method_config'] = encode(dict(imputer=asdict(configs[method+'_baseline_config']) if method != 'mean' else {'strategy':'observed_training_mean'},
                                                   vb=asdict(configs['dvai_cholesky_gvb_config']),
                                                   seeds=seeds, prediction_theta_mc=PREDICTION_THETA_MC))
        progress['actual_method_config'] = base['actual_method_config']
        print(method+' actual configuration: '+base['actual_method_config'],flush=True)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            out = run_baseline(method, rawtr, rawte, ytr, yte, masks, configs,
                               seeds, torch.device(args.device), transform)
        progress.update(completed_k=out.get('actual_k',1), attempted_k=out.get('attempted_k',1))
        notes += [str(w.message) for w in captured]
        if out.get('warning'): notes.append(out['warning'])
        notes = list(dict.fromkeys(notes))
        p = out.get('p_test')
        extra_prediction_sec = 0.
        if p is None:
            sync(args.device)
            pred_start = time.perf_counter()
            p = predict(out['X_test_imp'], out['mu'], out['cov'], seeds['prediction'])
            sync(args.device)
            extra_prediction_sec = time.perf_counter()-pred_start
        row = dict(base, row_type='final', k=1, actual_k=1, attempted_k=1, is_final=True, stop_reason=out.get('stop_reason','completed'),
                   status=out.get('status','completed'), warning='; '.join(notes) or None,
                   derived_seeds=encode(seeds), prediction_sec=extra_prediction_sec,
                   training_sec=out.get('training_sec'), imputation_sec=out.get('imputation_sec'),
                   inference_sec=out.get('inference_sec'),
                   train_time_sec=out['train_time_sec'], test_time_sec=out['test_time_sec']+extra_prediction_sec,
                   total_time_sec=out['total_time_sec']+extra_prediction_sec, diagnostic_stage=out['diagnostic_stage'],
                   timing_scope=out.get('timing_scope','separate measured stages; non-applicable timings blank'))
        row['prediction_sec'] += out.get('prediction_sec',0.)
        row['postfit_diagnostic_sec'] = out.get('diagnostic_sec')
        diagnostic_total += out.get('diagnostic_sec',0.)
        for field,value in list(out.items()):
            if isinstance(value, (dict,list)) and field=='training_diagnostics':
                row[field] = encode(value)
        row.update({k:v for k,v in out.items() if k.startswith('miwae_') or k in {'implementation_note','runtime_warning','imputation_estimator','posterior_available'}})
        for field in list(row):
            if isinstance(row[field],(dict,list)): row[field] = encode(row[field])
        progress['failure_stage'] = 'prediction_and_metrics'
        add_estimate(row, out['theta_estimate'], out['posterior_cov'], reference, p, yte, args.save_predictions, posterior_available=out['posterior_available'])
        row.update(actual_k=out.get('actual_k',1), k=out.get('actual_k',1), attempted_k=out.get('attempted_k',1),
                   baseline_diagnostics=encode({k:v for k,v in out.items() if k in {'outer_history','gvb_fits','regression_input_shape','num_imputations','mice_iterations','missforest_iter','missforest_delta'}}))
        rows.append(metrics(row, out['X_train_imp'], out['X_test_imp'], out.get('X_test_samples')))
        training_samples = out.get('X_train_samples')
        if training_samples is None:
            training_samples = out['X_train_imp']
        add_training_evaluation(row, xtr, out['X_train_imp'], masks[0], training_samples,
                                ytr, out['theta_estimate'], out['posterior_cov'],
                                seed('train_prediction'), args.save_predictions)
        return
    tracker = StopTracker(args.posterior_tol)
    previous = None
    train_total = test_total = 0.
    for k in range(1, args.K_max + 1):
        progress.update(attempted_k=k, failure_stage='diffusion_training')
        seeds = {p: seed(p, k) for p in ['network_init', 'training', 'sampling_train', 'sampling_test', 'langevin_train', 'langevin_test', 'vb', 'prediction', 'vb_monitor']}
        dc = deepcopy(configs['diff_config'])
        vc = deepcopy(configs['dvai_cholesky_gvb_config'])
        dc.init_seed = seeds['network_init']
        dc.train_seed = seeds['training']
        vc.seed = seeds['vb']
        vc.track_history = True
        base['actual_method_config'] = encode(dict(diffusion=asdict(dc), vb=asdict(vc), J=base['J'],
                                                   stop=cfg.dvai_stop_protocol(), seeds=seeds))
        progress['actual_method_config'] = base['actual_method_config']
        with measure(args.device) as measured:
            imputer = ConditionalDiffusionImputer(10, dc, torch.device(args.device))
            info = imputer.fit(transform.to_network(ztr))
        training = training_actual = measured['seconds']
        t = time.perf_counter()
        progress['failure_stage'] = 'train_sampling'
        set_global_seed(seeds['sampling_train'])
        train_sampling = {}
        samples = imputer.sample(transform.to_network(transform.standardize(xtr, masks[0])), masks[0], base['J'], timings=train_sampling, langevin_seed=seeds['langevin_train'])
        samples = transform.unstandardize(transform.from_network(samples), xtr, masks[0])
        current = samples.mean(0)
        current[~masks[0]] = xtr[~masks[0]]
        sync(args.device)
        imputation = time.perf_counter() - t - train_sampling.get('diagnostic_sec', 0.)
        row = dict(base, row_type='iteration', k=k, actual_k=k, attempted_k=k, derived_seeds=encode(seeds), diffusion_diagnostics=encode(info), training_sec=training, shared_training_sec=training, training_actual_sec=training_actual, train_conditional_sampling_sec=train_sampling['conditional_sampling_sec'], imputation_sec=imputation, inference_sec=None, prediction_sec=None, diffusion_loss=info['best_loss'], train_langevin_sec=train_sampling['langevin_sec'], prediction_completion_samples=base['J'], likelihood_completion_samples=base['J'], vb_input='completed_samples', method_detail='conditional_samples_logmeanexp_mc_vb')
        distance = None
        progress['failure_stage'] = 'vb_fit'
        vb_info = {}
        with measure(args.device) as measured:
            mean, cov = fit_vb(samples, ytr, vc, args.device, vb_info, save_elbo_history=True)
        row['inference_sec'] = measured['seconds']
        row['vb_diagnostics'] = encode(vb_info)
        stop_started = time.perf_counter()
        distance = posterior_components(*previous, mean, cov)['db_per_dim'] if previous is not None else None
        previous = (mean, cov)
        should_stop = tracker.update(k, distance)
        stopping_sec = time.perf_counter()-stop_started
        final = k == args.K_max or (should_stop and args.actual_early_stop)
        row.update(convergence_distance=distance, consecutive=tracker.streak, first_stop_k=tracker.first_stop_k, stop_reason='would_stop' if should_stop else 'continue')
        progress['completed_k'] = k
        if not final:
            t = time.perf_counter()
            ztr = ((current - transform.center) / transform.std).astype(np.float32)
            train_total += training_actual + imputation + row['inference_sec'] + time.perf_counter()-t
            progress.update(train_time_sec=train_total, test_time_sec=test_total)
            row.update(train_time_sec=train_total, test_time_sec=0., total_time_sec=train_total,
                       posterior_available=True, posterior_mean=encode(mean), posterior_cov=encode(cov),
                       stopping_sec=stopping_sec, timing_scope=SENSITIVITY_TIMING_PROTOCOL,
                       timing_protocol=SENSITIVITY_TIMING_PROTOCOL)
            rows.append(row)
            print(f'  rep={rep} {candidate["name"]} k={k} DB/p={distance} continue', flush=True)
            continue
        progress['failure_stage'] = 'test_sampling'
        t = time.perf_counter()
        set_global_seed(seeds['sampling_test'])
        test_sampling = {}
        te_samples = imputer.sample(transform.to_network(zte), masks[1], base['J'], timings=test_sampling, langevin_seed=seeds['langevin_test'])
        row.update(test_conditional_sampling_sec=test_sampling['conditional_sampling_sec'], test_langevin_sec=test_sampling['langevin_sec'])
        te_samples = transform.unstandardize(transform.from_network(te_samples), xte, masks[1])
        sync(args.device)
        rawte = te_samples.mean(0)
        rawte[~masks[1]] = xte[~masks[1]]
        row['test_sampling_sec'] = time.perf_counter() - t - test_sampling.get('diagnostic_sec', 0.)
        row['imputation_sec'] += row['test_sampling_sec']
        train_total += training_actual + imputation + row['inference_sec']
        test_total += row['test_sampling_sec']
        progress['failure_stage'] = 'final_prediction'
        with measure(args.device) as measured:
            p = predict(te_samples, mean, cov, seeds['prediction'], PREDICTION_THETA_MC)
        row['prediction_sec'] = measured['seconds']
        test_total += row['prediction_sec']
        progress.update(train_time_sec=train_total, test_time_sec=test_total, failure_stage='final_evaluation')
        stats = imputation_error_stats(xte, rawte, masks[1], np.sqrt(np.diag(SIGMA)),
                                       'test', samples=te_samples, independent=True)
        score = {key:stats[key] for key in ('test_impute_rmse', 'test_impute_mae', 'test_impute_crps',
                                           'test_crps_type', 'test_crps_reason')}
        prediction_scores = classification_metrics(yte, p)
        score.update({key:prediction_scores[key] for key in ('AUC', 'ACC', 'Brier', 'AUC_reason')})
        overlap = posterior_components(mean, cov, *reference)['overlap']
        add_training_evaluation(row, xtr, current, masks[0], samples, ytr, mean, cov,
                                seed('train_prediction', k), args.save_predictions)
        if score['test_crps_type'] != 'fair_samples' or not np.isfinite([overlap, score['test_impute_rmse'], score['test_impute_crps']]).all():
            raise ValueError('Final Overlap, RMSE and fair CRPS must be finite')
        reason = 'converged' if args.actual_early_stop and should_stop else 'max_iter'
        rows.append(dict(row, **score, overlap=overlap, row_type='final', is_final=True,
                         status='completed', stop_reason=reason, train_time_sec=train_total,
                         test_time_sec=test_total, total_time_sec=train_total+test_total,
                         posterior_mean=encode(mean), posterior_cov=encode(cov),
                         theta_estimate=encode(mean), posterior_available=True,
                         test_probabilities=encode(p) if args.save_predictions else None,
                         test_labels=encode(yte) if args.save_predictions else None,
                         stopping_sec=stopping_sec,
                         timing_protocol=SENSITIVITY_TIMING_PROTOCOL,
                         timing_scope=SENSITIVITY_TIMING_PROTOCOL))
        break
    for row in rows:
        row['first_stop_k'] = tracker.first_stop_k
        row['final_stop_reason'] = rows[-1]['stop_reason']

def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def validate_scenarios(scenarios):
    if not isinstance(scenarios, (list, tuple)) or not scenarios:
        raise ValueError('Provide a nonempty explicit SCENARIOS list of (mechanism, missing_rate) pairs')
    pairs = []
    for item in scenarios:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f'Invalid scenario pair: {item!r}')
        mechanism, rate = item
        if mechanism not in ('MCAR', 'MAR', 'MNAR') or isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise ValueError(f'Invalid scenario pair: {item!r}')
        rate = validate_missing_rate(mechanism, rate)
        if mechanism == "MAR":
            mar_target_rate(D, rate)
        pair = (mechanism, float(rate))
        if pair in pairs:
            raise ValueError(f'Duplicate scenario pair: {pair!r}')
        pairs.append(pair)
    return pairs


def run_experiment(args, candidates=None):
    stop_protocol = cfg.dvai_stop_protocol(args.posterior_tol, k_max=args.K_max)
    threads = configure_single_thread()
    comparison = getattr(args, 'compare_three', False)
    paired_comparison = getattr(args, 'compare_mcar_mar50', False)
    terminal_comparison = comparison or paired_comparison
    args.out_dir = str(result_directory('testoutput' if terminal_comparison else 'simulation'))
    if paired_comparison and (comparison or args.stage != 'formal' or args.master_seed != 2026 or args.rep_start != 0 or
                              args.reps != 1 or args.J != cfg.dvai_config.J or
                              args.methods != ['dvai','mean','mice','missforest','miwae','diffputer'] or
                              args.mechanisms != ['MCAR','MAR'] or args.rates != [.5] or args.retry_failed or args.save_predictions):
        raise ValueError('Paired comparison requires formal seed2026 rep0, current formal J, six ordered methods, MCAR/MAR50, no retry or saved predictions')
    if comparison and (args.stage != 'formal' or args.master_seed != 2026 or args.rep_start != 0 or
                       args.reps != 1 or args.J != 20 or args.methods != ['dvai','mice','missforest'] or
                       args.mechanisms != ['MAR'] or args.rates != [.5] or args.retry_failed or args.save_predictions):
        raise ValueError('Three-method comparison requires formal seed2026 rep0 J20, methods dvai/mice/missforest, MAR50, no retry or saved predictions')
    if args.stage == 'formal' and args.K_max != 10:
        raise ValueError('Formal DVAI K_max must be 10')
    if args.D != 10:
        raise ValueError('Fixed population requires D=10')
    if terminal_comparison and (args.n_train != 700 or args.n_test != 300):
        raise ValueError('Comparison requires 700 train / 300 test')
    if args.n_train < 2 or args.n_test < 1 or args.reps < 1 or args.K_max < 1:
        raise ValueError('Invalid experiment size')
    if 3 * args.n_train != 7 * args.n_test:
        raise ValueError('Simulation requires a 7:3 train/test split')
    args.device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if not set(args.methods) <= set(METHODS):
        raise ValueError('Unknown methods')
    if hasattr(args, 'scenarios'):
        scenarios = validate_scenarios(args.scenarios)
    elif getattr(args, 'experiment', None) == 'check':
        raise ValueError('Confirmation requires explicit scenario pairs')
    else:
        scenarios = validate_scenarios([(m, r) for m in args.mechanisms for r in args.rates])
    candidates = candidates or [dict(name='locked' if args.stage == 'formal' else 'initial',
                                  J=args.J)]
    for c in candidates:
        validate_candidate(c)
    if len({c['name'] for c in candidates}) != len(candidates):
        raise ValueError('Duplicate candidate names')
    args.protocol_version = PROTOCOL_VERSION
    args.metric_scale = METRIC_SCALE
    args.metric_protocol = METRIC_PROTOCOL
    args.crps_protocol = CRPS_PROTOCOL
    args.timing_protocol = TIMING_PROTOCOL
    args.dvai_timing_protocol = SENSITIVITY_TIMING_PROTOCOL
    args.dvai_evaluation = 'final_returned_round_only'
    args.coordinate_protocol = 'public Z=(X-center)/std; diffusion U=Z/2; regression X'
    args.sampling_settings = sampling_fields()
    set_global_seed(args.master_seed)
    args.determinism = dict(algorithms=torch.are_deterministic_algorithms_enabled(),
                            cudnn=torch.backends.cudnn.deterministic, benchmark=torch.backends.cudnn.benchmark,
                            matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32)
    configs = local_configs(args)
    if paired_comparison:
        print(encode(dict(generation=design_parameters(), J=args.J,
                          diffusion=asdict(configs['diff_config']), stop=stop_protocol,
                          methods=args.methods, scenarios=scenarios, device=args.device,
                          prediction_theta_mc=PREDICTION_THETA_MC, timing_protocol=TIMING_PROTOCOL)),flush=True)
    if comparison:
        print(encode(dict(generation=design_parameters(), response='Bernoulli(expit(raw_X @ THETA)); no intercept',
                          configs={k:asdict(configs[k]) for k in ('diff_config','dvai_cholesky_gvb_config','mice_baseline_config','missforest_baseline_config')},
                          prediction=dict(theta_mc=PREDICTION_THETA_MC, ACC_threshold=.5,
                                          aggregation='probabilities averaged over theta and then completion samples'),
                          timing_protocol=TIMING_PROTOCOL)), flush=True)
    if args.stage == "formal" and configs["diffputer_baseline_config"].K != 10:
        raise ValueError("Formal DiffPuter requires K=10")
    dependencies = runtime_identity(args.device)
    entry = 'simulation_main.py'
    version = code_version(entry)
    full_version = code_version()
    effective = dict(vars(args), candidates=candidates, stop=stop_protocol,
                     configs={k:asdict(v) for k,v in configs.items()}, prior_var=5.,
                     generation_parameters=design_parameters())
    metadata = dict(stop_protocol=encode(stop_protocol), missingness_model='ignored',stage=args.stage, data_stage=args.stage, screening_stage=getattr(args,'experiment',None),
                    distribution='single_gaussian', generation_parameters=encode(design_parameters()), n_total=args.n_train+args.n_test, n_train=args.n_train,
                    n_test=args.n_test, D=10, master_seed=args.master_seed, theta_true=encode(THETA),
                    prior_var=5., regression_intercept=False, effective_config=encode(effective),
                    code_version=encode(full_version), protocol_source=encode(version),
                    overlap_definition='exp(-DB/p)', metric_scale=METRIC_SCALE,
                    metric_protocol=METRIC_PROTOCOL, crps_protocol=CRPS_PROTOCOL, timing_protocol=TIMING_PROTOCOL,
                    evaluation_std=encode(np.sqrt(np.diag(SIGMA))),
                    protocol_revision=PROTOCOL_VERSION, thread_settings=encode(threads),
                    dependency_versions=encode(dependencies), device=args.device,
                    device_name=torch.cuda.get_device_name() if args.device.startswith('cuda') else 'CPU',
                    covariance_encoding='JSON full matrix, original feature order',
                    prediction_theta_mc=PREDICTION_THETA_MC, vb_theta_mc=configs['dvai_cholesky_gvb_config'].num_mc,
                    test_sample_ids=encode(np.arange(args.n_train,args.n_train+args.n_test)))
    if paired_comparison:
        metadata.update(parameter_order=encode([f'X{i}' for i in range(1,D+1)]),
                        regression_coordinates='raw_X_no_intercept')
    sample_suffix = f'_N{args.n_train + args.n_test}'
    paths = {(mechanism, rate): Path(args.out_dir) /
             ('compare_mcar_mar50.csv' if paired_comparison else 'compare_dvai_mice_missforest.csv' if comparison else f'Simulation_{mechanism}_{round(rate * 100):03d}{sample_suffix}.csv')
             for mechanism, rate in scenarios}
    outputs = list(dict.fromkeys(paths.values()))
    if not outputs:
        raise ValueError('At least one mechanism and missing rate is required')
    if len(outputs) != len(paths) and not paired_comparison:
        raise ValueError('Missing rates collide at integer percentage filename precision')
    if comparison and outputs[0].exists():
        previous = read_result_rows(outputs[0])
        raise ValueError(f'Comparison output already exists; no overwrite or automatic rerun. '
                         f'Rows={len(previous)}, protocols={sorted({r.get("protocol_id", "") for r in previous})}')
    with ExitStack() as stack:
        stores = {}
        for path in outputs:
            identity = protocol_identity(effective, [s for s in scenarios if paths[s] == path])
            protocol = hashlib.sha256(encode(dict(effective=identity, code=version['source_sha256'], dependencies=dependencies)).encode()).hexdigest()
            stores[path] = stack.enter_context(ResultCSV(path, protocol, protocol_details=identity))
        rep_start = getattr(args,'rep_start',0)
        if rep_start < 0: raise ValueError('rep_start must be nonnegative')
        run_id = uuid.uuid4().hex
        for rep in range(rep_start, rep_start+args.reps):
            sync(args.device)
            stream = lambda purpose, *keys: derive_seed(args.master_seed, args.stage, rep, purpose, *keys)
            xtr, xte, ytr, yte, data_seeds = repetition_data(args, rep)
            data_seeds['reference_monitor'] = stream('reference_monitor')
            prepared_scenarios = {}
            for mechanism, rate in scenarios:
                store = stores[paths[mechanism, rate]]
                prep_id = encode([rep,mechanism,rate,'data_preparation'])
                if store.complete(prep_id,retry_failed=getattr(args,'retry_failed',False)):
                    continue
                try:
                    prepared_scenarios[mechanism,rate] = prepare_scenario(
                        args,rep,mechanism,rate,xtr,xte,configs)
                except Exception as exc:
                    message = f'dataset=simulation rep={rep} mechanism={mechanism} target_rate={rate}: {type(exc).__name__}: {exc}'
                    store.commit(prep_id,[dict(metadata,row_type='failure',method='shared_data_preparation',
                        rep=rep,mechanism=mechanism,target_missing_rate=rate,status='failed',
                        is_final=True,actual_k=0,attempted_k=0,stop_reason='failed',
                        failure_stage='data_preparation',warning=message,data_seeds=encode(data_seeds))])
                    report_saved(store, mechanism, rate, rep, 'shared_data_preparation', 'failed')
                    print(message,flush=True)
            if not prepared_scenarios:
                continue
            ref_identity = full_vb_identity(xtr, ytr, data_seeds, configs, args.device)
            ref_meta = dict(reference_identity=encode(ref_identity),
                            reference_id=hashlib.sha256(encode(ref_identity).encode()).hexdigest())
            reference_id = f'reference/{rep}'
            saved = next((r for store in stores.values() for r in store.rows
                          if r.get('trajectory') == reference_id and r.get('row_type') == 'reference'),None)
            historical = None
            if comparison and saved is None:
                started = time.perf_counter()
                historical = comparison_reference(xtr, ytr, data_seeds, configs, args.device)
                reference_load_sec = time.perf_counter()-started
                if historical is not None:
                    saved = dict(metadata, row_type='reference', method='complete_reference', rep=rep,
                                 status='completed', reference_mean=historical['reference_mean'],
                                 reference_cov=historical['reference_cov'], reference_id=historical['reference_id'],
                                 reference_identity=historical['reference_identity'],
                                 reference_source=str(sensitivity_path('sigma', 'MAR')),
                                 reference_source_protocol=historical['protocol_id'],
                                 reference_historical_fit_sec=float(historical['reference_time_sec']),
                                 reference_historical_timing_protocol=historical.get('reference_timing_protocol', 'legacy_unspecified'),
                                 reference_fit_sec_current=0., reference_load_sec=reference_load_sec,
                                 inference_sec=0., reference_reused=True, data_seeds=encode(data_seeds),
                                 numerical_warnings=historical.get('numerical_warnings',''))
            if not saved and not getattr(args,'retry_failed',False):
                failed_store = next((store for store in stores.values() if any(r.get('trajectory')==reference_id and r.get('row_type')=='failure' for r in store.rows)),None)
                if failed_store is not None and not paired_comparison:
                    failed_store.complete(reference_id)
                    continue
                if failed_store is not None and paired_comparison:
                    saved = failed_store.current_rows(reference_id)[-1]
            reference_was_reused = saved is not None
            if saved:
                if paired_comparison and saved.get('status') == 'failed':
                    if saved.get('reference_identity') != encode(ref_identity):
                        raise ValueError('Failed Full-VB record has incompatible fitting identity')
                    reference = None
                else:
                    reference = validate_full_vb(saved, ref_identity)
                if comparison:
                    sync(args.device); started = time.perf_counter()
                    ref_p = predict(xte,*reference,data_seeds['reference_prediction'])
                    sync(args.device); prediction_sec = time.perf_counter()-started
                    scores = classification_metrics(yte,ref_p)
                    saved.update(reference_AUC=scores['AUC'],reference_ACC=scores['ACC'],reference_Brier=scores['Brier'],
                                 reference_AUC_reason=scores['AUC_reason'],reference_prediction_sec=prediction_sec,
                                 reference_prediction_seed=data_seeds['reference_prediction'])
            else:
                vc = deepcopy(configs['dvai_cholesky_gvb_config']); vc.seed = data_seeds['reference_vb']
                if paired_comparison:
                    print(f'Full-VB rep={rep}: one fresh shared reference fit',flush=True)
                reference_info = {}
                failure_stage = 'reference_vb'
                try:
                    with warnings.catch_warnings(record=True) as reference_warnings:
                        warnings.simplefilter('always')
                        with measure(args.device) as reference_timing:
                            reference = fit_vb(xtr,ytr,vc,args.device,reference_info)
                    inference_sec = reference_timing['seconds']
                    failure_stage = 'reference_prediction'
                    pred_start = time.perf_counter()
                    ref_p = predict(xte,*reference,data_seeds['reference_prediction'])
                except Exception as exc:
                    if not paired_comparison:
                        for store in stores.values():
                            store.commit(reference_id,[dict(metadata,row_type='failure',method='complete_reference',rep=rep,
                                status='failed',is_final=True,actual_k=0,attempted_k=0,failure_stage=failure_stage,
                                warning=f'{type(exc).__name__}: {exc}',data_seeds=encode(data_seeds))])
                            print(f'[saved] reference | rep={rep} | status=failed | file={store.path}', flush=True)
                        continue
                    reference = None
                    saved = dict(metadata, **ref_meta, row_type='failure', method='complete_reference', rep=rep,
                                 status='failed', is_final=True, actual_k=0, attempted_k=0, stop_reason='failed',
                                 failure_stage=failure_stage, warning=f'{type(exc).__name__}: {exc}',
                                 data_seeds=encode(data_seeds), inference_sec=None,
                                 overlap=None, overlap_reason='Full-VB reference failed',
                                 reference_mean=None, reference_cov=None)
                    print('Full-VB FAILED; continue without Overlap: '+saved['warning'],flush=True)
                else:
                    sync(args.device)
                    reference_prediction_sec = time.perf_counter()-pred_start
                    ref_scores = classification_metrics(yte,ref_p)
                    saved = dict(metadata, **ref_meta, reference_AUC=ref_scores['AUC'],
                                 reference_ACC=ref_scores['ACC'], reference_Brier=ref_scores['Brier'], reference_prediction_seed=data_seeds['reference_prediction'],
                                 reference_AUC_reason=None if len(np.unique(yte))==2 else 'single_class_test_labels',
                                 reference_prediction_sec=reference_prediction_sec, vb_diagnostics=encode(reference_info),
                                 row_type='reference', method='complete_reference', rep=rep,
                                 numerical_warnings='; '.join(str(w.message) for w in reference_warnings),
                                 reference_mean=encode(reference[0]), reference_cov=encode(reference[1]),
                                 data_seeds=encode(data_seeds), inference_sec=inference_sec, reference_timing_protocol=TIMING_PROTOCOL, status='completed')
            if reference is not None:
                add_estimate(saved, *reference, reference, None, yte)
            if paired_comparison and reference is not None:
                saved.update(train_time_sec=float(saved['inference_sec']), test_time_sec=float(saved['reference_prediction_sec']),
                             total_time_sec=float(saved['inference_sec'])+float(saved['reference_prediction_sec']),
                             AUC=saved['reference_AUC'],ACC=saved['reference_ACC'],Brier=saved['reference_Brier'],
                             actual_k=1,is_final=True,stop_reason=json.loads(saved['vb_diagnostics'])['inner_stop_reason'])
                saved['vb_diagnostics'] = encode({k:v for k,v in json.loads(saved['vb_diagnostics']).items() if not k.endswith('_history')})
                print('Full-VB RESULT '+encode({k:saved.get(k) for k in ('status','AUC','ACC','Brier','train_time_sec','test_time_sec','total_time_sec','stop_reason','numerical_warnings')}),flush=True)
            if comparison:
                saved.setdefault('reference_fit_sec_current',saved['inference_sec'])
                saved.setdefault('reference_reused',False)
                print('Full-VB '+encode({k:saved.get(k) for k in ('reference_reused','reference_historical_fit_sec',
                      'reference_fit_sec_current','reference_load_sec','reference_prediction_sec',
                      'reference_AUC','reference_ACC','reference_Brier')}),flush=True)
            for store in stores.values():
                if not any(r.get('trajectory') == reference_id and r.get('row_type') in ({'reference','failure'} if paired_comparison else {'reference'}) for r in store.rows):
                    store.commit(reference_id, [dict(saved)])
                    print(f'[saved] reference | rep={rep} | status={saved["status"]} | file={store.path}', flush=True)
            for mechanism, rate in scenarios:
                if (mechanism,rate) not in prepared_scenarios:
                    continue
                store = stores[paths[mechanism,rate]]
                masks,transform,scenario_metadata,mask_seeds = prepared_scenarios[mechanism,rate]
                current_data_id = data_digest(xtr,xte,ytr,yte,*masks,transform.center,transform.std)
                if historical is not None:
                    expected = dict(data_id=current_data_id,preprocess_center=encode(transform.center),
                                    preprocess_std=encode(transform.std),mask_seeds=encode(mask_seeds))
                    for key,value in expected.items():
                        if historical.get(key) != value:
                            raise ValueError(f'Historical scenario identity differs at {key}; no method fitting')
                    scenario_metadata.update(reference_id=historical['reference_id'],reference_identity=historical['reference_identity'])
                scenario_metadata.update(data_id=current_data_id,regression_coordinates='raw_X_no_intercept')
                print(encode(dict(mechanism=mechanism, target_rate=rate, data_id=current_data_id, train_missing=float(masks[0].mean()),
                      test_missing=float(masks[1].mean()), train_observed=(~masks[0]).sum(0),
                      test_observed=(~masks[1]).sum(0), train_positive=float(ytr.mean()),test_positive=float(yte.mean()))),flush=True)
                for candidate in candidates:
                    for method in args.methods:
                        trajectory = encode([rep,mechanism,rate,candidate['name'],method])
                        for existing in store.current_rows(trajectory):
                            if existing.get('data_id') != scenario_metadata['data_id']:
                                raise ValueError(f'Saved comparison data/mask/preprocessing identity differs: {trajectory}')
                            validate_terminal_record(existing)
                        if store.complete(trajectory, retry_failed=getattr(args,'retry_failed',False)):
                            print(f'Skip existing {trajectory}', flush=True)
                            continue
                        rows = []
                        print(f'Run {trajectory}',flush=True)
                        meta = dict(metadata,**scenario_metadata,data_seeds=encode(data_seeds),mask_seeds=encode(mask_seeds),
                                    reference_inference_sec=float(saved['inference_sec']) if saved.get('inference_sec') is not None else None,
                                    reference_reused=reference_was_reused, run_id=run_id,
                                    reference_prediction_sec=saved.get('reference_prediction_sec'),
                                    reference_numerical_warnings=saved.get('numerical_warnings',''))
                        meta.update({key:saved.get(key) for key in ['reference_AUC','reference_ACC','reference_prediction_seed','reference_AUC_reason']})
                        meta.update(ref_meta)
                        progress = dict(attempted_k=0, failure_stage='initialization')
                        with collect_training() as events, warnings.catch_warnings(record=True) as captured:
                            warnings.simplefilter('always')
                            try:
                                run_trajectory(args,candidate,method,rep,mechanism,rate,xtr,xte,ytr,yte,masks,
                                               reference,deepcopy(configs),meta,rows,progress,transform)
                                validate_terminal_record(rows[-1])
                            except Exception as exc:
                                for incomplete in rows:
                                    if incomplete.get('row_type') == 'final':
                                        incomplete.update(row_type='invalid_result', is_final=False, status='invalid')
                                row = dict(meta, row_type='failure', status='failed', stop_reason='failed', is_final=True,
                                           actual_k=progress.get('completed_k',max((r.get('k',0) for r in rows),default=0)),
                                           **{key:value for key,value in progress.items() if key != 'completed_k'},
                                           method=method,candidate=candidate['name'],rep=rep,mechanism=mechanism,
                                           target_missing_rate=rate, J=candidate['J'],
                                           train_actual_missing_rate=float(masks[0].mean()),
                                           test_actual_missing_rate=float(masks[1].mean()),
                                           posterior_available=False, **sampling_fields(configs['diff_config'].sample_steps, configs['diff_config']), warning=f'{type(exc).__name__}: {exc}')
                                if method == 'diffputer':
                                    row.update(K_max=configs['diffputer_baseline_config'].K)
                                    row.update(getattr(exc, 'baseline_progress', {}))
                                    row['k'] = row['actual_k']
                                rows.append(row)
                                print(row['warning'],flush=True)
                            if rows:
                                rows[-1]['training_diagnostics'] = encode(events)
                                rows[-1]['numerical_warnings'] = '; '.join(dict.fromkeys(str(w.message) for w in captured))
                        if terminal_comparison:
                            for row in rows:
                                if method != 'dvai' and row.get('vb_diagnostics'):
                                    info = json.loads(row['vb_diagnostics'])
                                    row['vb_diagnostics'] = encode({k:v for k,v in info.items() if not k.endswith('_history')})
                            final = rows[-1]
                            if method == 'dvai':
                                final['outer_diagnostics'] = encode([{k:r.get(k) for k in ('k','convergence_distance','consecutive','stop_reason','diffusion_diagnostics','vb_diagnostics')} for r in rows if r.get('row_type') in ('iteration','final')])
                            rows = [final]
                        store.commit(trajectory,rows)
                        report_saved(store, mechanism, rate, rep, method, rows[-1]['status'])
                        if terminal_comparison:
                            print('RESULT '+encode({k:final.get(k) for k in ('method','status','actual_k','stop_reason',
                                   'train_impute_rmse','train_impute_mae','train_AUC',
                                   'test_impute_rmse','test_impute_mae','test_impute_crps','test_crps_type','overlap','AUC','ACC','Brier',
                                  'train_time_sec','test_time_sec','total_time_sec','warning','numerical_warnings')}),flush=True)
    for output in outputs:
        print(f'Saved {output}')
    return outputs[0] if len(outputs) == 1 else outputs


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['smoke','formal'],default='formal')
    p.add_argument('--master-seed',type=int,default=2026)
    p.add_argument('--rep-start',type=int,default=0)
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--D',type=int,default=10)
    p.add_argument('--n-total',type=int,default=argparse.SUPPRESS,
                   help='Total samples with a 7:3 train/test split; use instead of --n-train/--n-test')
    p.add_argument('--n-train',type=int)
    p.add_argument('--n-test',type=int)
    p.add_argument('--reps',type=int, help='Number of rep indices in [rep-start, rep-start + reps); completed indices are skipped')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--mechanisms',nargs='+',choices=['MCAR','MAR','MNAR'],default=['MCAR','MAR'])
    p.add_argument('--rates',nargs='+',type=float,default=[.3,.5,.7],
                   help='Missing fractions, e.g. 0.1 0.2 (default: 0.3 0.5 0.7)')
    p.add_argument('--J',type=int,choices=[5,10,20,40],default=cfg.dvai_config.J)
    p.add_argument('--posterior-tol',type=float,choices=[cfg.dvai_config.posterior_tol],default=cfg.dvai_config.posterior_tol)
    p.add_argument('--device',default='auto')
    p.add_argument('--save-predictions',action='store_true',help='Also retain final train/test probabilities and labels')
    p.add_argument('--compare-three',action='store_true',help='One formal rep0 MAR50 DVAI/MICE/MissForest comparison in testoutput using current DVAI defaults')
    p.add_argument('--compare-mcar-mar50',action='store_true',help='One formal rep0 six-method MCAR50/MAR50 comparison using current DVAI defaults; terminal records in one CSV')
    return p


def main(argv=None):
    parser=build_parser(); args=parser.parse_args(argv)
    n_total = vars(args).pop('n_total', None)
    if n_total is not None:
        if n_total < 10 or n_total % 10:
            parser.error('--n-total must be a positive multiple of 10 for the 7:3 split')
        if args.n_train is not None or args.n_test is not None:
            parser.error('Use --n-total or --n-train/--n-test, not both')
        args.n_train, args.n_test = 7 * n_total // 10, 3 * n_total // 10
    for key,value in dict(J=20,n_train=700,
                          n_test=300,reps=1 if args.stage=='smoke' else 50).items():
        if getattr(args,key) is None: setattr(args,key,value)
    args.K_max=3 if args.stage=='smoke' else cfg.dvai_config.K
    args.actual_early_stop=True
    run_experiment(args)


if __name__=='__main__':
    main()
