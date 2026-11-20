import argparse
import hashlib
import importlib.metadata
import json
import sys
import warnings
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np
import torch
import simulation_main as sim
from configs import dvai_config, dvai_stop_protocol
from generate_miss.fixed_design import MU, STD, SIGMA, THETA, design_parameters
from generate_miss.mask_mechanisms import missingness_parameters
from untils.metrics import CRPS_PROTOCOL
from untils.result_io import ResultCSV, sensitivity_path, source_version
from untils.time import measure, TIMING_PROTOCOL, SENSITIVITY_TIMING_PROTOCOL
from models.diffusion_imputer import DIFFUSION_TRAINING_PROTOCOL
from untils.seed_device import configure_single_thread, set_global_seed

SIGMAS = (.1, .2, .4, .8)
L_VALUES = (2, 4, 6, 8)
FIXED_J = 20
ETA = .02
REVISION = 'single_stage_fixedJ20_sigmaL16_{mechanism}50_final_prediction_v5'
TIMING = SENSITIVITY_TIMING_PROTOCOL
METRICS = 'joint_gaussian_overlap_final_rmse_mae_fair_crps_auc_acc_v4'
SCORES = ('overlap', 'test_impute_rmse', 'test_impute_mae', 'test_impute_crps', 'AUC', 'ACC')
SUCCESS = 'completed'


def digest(value):
    return hashlib.sha256(sim.encode(value).encode()).hexdigest()


def candidate(sigma, L):
    if sigma not in SIGMAS or L not in L_VALUES:
        raise ValueError('Unsupported study configuration')
    return dict(J=FIXED_J, sigma_L=sigma, L=L, eta=ETA)


def candidate_id(c):
    return f'J{c["J"]}_sigma{c["sigma_L"]:.2f}_L{c["L"]}'


def task_id(rep, c):
    return f'dvai/{rep}/{candidate_id(c)}'


def grid_candidates():
    return [candidate(sigma, L) for sigma in SIGMAS for L in L_VALUES]


def active_record(store, identity):
    rows = store.current_rows(identity)
    if len(rows) > 1:
        raise ValueError(f'Duplicate active result: {identity}')
    if rows and rows[0].get('protocol_id') != store.fingerprint:
        raise ValueError(f'Conflicting result protocol: {identity}')
    if rows and rows[0].get('row_type') == 'final' and rows[0].get('status') == SUCCESS and not successful(rows[0]):
        raise ValueError(f'Invalid successful result (six scores/timing/stopping required): {identity}')
    return rows[0] if rows else None


def successful(row):
    if not row or row.get('status') != SUCCESS or row.get('row_type') != 'final':
        return False
    try:
        times = [float(row[k]) for k in ('train_time_sec', 'test_time_sec', 'total_time_sec')]
        return bool(row.get('test_crps_type') == 'fair_samples'
                    and np.isfinite([float(row[k]) for k in SCORES] + times).all()
                    and 0 <= float(row['overlap']) <= 1
                    and 0 <= float(row['AUC']) <= 1 and 0 <= float(row['ACC']) <= 1
                    and min(float(row['test_impute_rmse']), float(row['test_impute_mae']), *times) >= 0
                    and np.isclose(times[2], times[0]+times[1])
                    and ((row.get('stop_reason') == 'converged'
                          and dvai_config.min_iter <= int(row['actual_k']) <= dvai_config.K)
                         or (row.get('stop_reason') == 'max_iter' and int(row['actual_k']) == dvai_config.K)))
    except (KeyError, ValueError, TypeError, OverflowError):
        return False


def study(device, mechanism='MAR'):
    if mechanism not in ('MAR', 'MCAR'):
        raise ValueError(f'Unsupported study mechanism: {mechanism}')
    args = SimpleNamespace(stage='formal', master_seed=2026, n_train=700, n_test=300,
                           D=10, K_max=dvai_config.K, posterior_tol=dvai_config.posterior_tol, actual_early_stop=True,
                           device=device, save_predictions=False, mechanism=mechanism)
    configs = {k:v for k,v in sim.local_configs(args).items() if k in ('diff_config','dvai_cholesky_gvb_config')}
    if configs['diff_config'].sample_steps != 50 or sim.StopTracker(args.posterior_tol).min_iter != 3 or sim.StopTracker(args.posterior_tol).patience != 2:
        raise ValueError('Sampling/stopping configuration conflicts with the approved study')
    if sim.PREDICTION_THETA_MC != 50:
        raise ValueError('Formal prediction count differs from the approved 50 parameter draws')
    if not (np.array_equal(MU, np.arange(1.,11.)) and np.array_equal(STD, MU)
            and np.array_equal(THETA, [.8,-.4,1.,0.,.6,-.5,.4,-.35,.2,-.18])
            and np.allclose(np.diag(SIGMA), STD**2)):
        raise ValueError('Current population conflicts with the approved single Gaussian design')
    np.linalg.cholesky(SIGMA)
    source = source_version('scripts/test/parameter_test.py')
    protocol = dict(revision=REVISION.format(mechanism=mechanism.lower()), args=vars(args), generation=design_parameters(),
                    missingness=missingness_parameters(mechanism, .5),
                    configs={k:asdict(v) for k,v in configs.items()}, prior_var=5.,
                    diffusion_training_protocol=DIFFUSION_TRAINING_PROTOCOL,
                    regression_coordinates='raw_X_no_intercept',
                    metric_protocol=METRICS, crps_protocol=CRPS_PROTOCOL,
                    prediction=dict(theta_mc=sim.PREDICTION_THETA_MC, acc_threshold=.5,
                                    samples='same_final_J_completions', integration='mean_sigmoid_then_mean_J',
                                    seed='existing_independent_prediction_stream', evaluation='final_returned_round_only'),
                    rmse=dict(field='test_impute_rmse', point='final_test_samples_mean_over_J',
                              coordinates='raw_X', scale='population_std', positions='test_missing',
                              historical_missing='unavailable_no_automatic_refit'),
                    selection_order=('mean_overlap_desc', 'mean_rmse_asc', 'L_asc', 'sigma_asc'),
                    evaluation_std=np.sqrt(np.diag(SIGMA)), timing_protocol=TIMING,
                    stop=dvai_stop_protocol(args.posterior_tol, k_max=args.K_max),
                    grids=dict(sigma=SIGMAS, L=L_VALUES, J=FIXED_J, eta=ETA),
                    seed_rule='existing formal derive_seed; candidate omitted',
                    source_sha256=source['source_sha256'],
                    versions={n:importlib.metadata.version(n) for n in ('torch','numpy','scipy','pandas','scikit-learn')},
                    device_name=torch.cuda.get_device_name() if device.startswith('cuda') else 'CPU')
    return args, configs, protocol, digest(protocol)


def prepare(args, rep, configs):
    xtr, xte, ytr, yte, seeds = sim.repetition_data(args, rep)
    masks, transform, scenario, mask_seeds = sim.prepare_scenario(args,rep,args.mechanism,.5,xtr,xte,configs)
    data_id = sim.data_digest(xtr, xte, ytr, yte, *masks, transform.center, transform.std)
    ref_identity = sim.full_vb_identity(xtr, ytr, seeds, configs, args.device)
    meta = dict(rep=rep, master_seed=args.master_seed, stage='formal', study=REVISION.format(mechanism=args.mechanism.lower()),
                distribution='single_gaussian', generation_parameters=sim.encode(design_parameters()),
                mechanism=args.mechanism, target_missing_rate=.5, missingness_parameters=sim.encode(missingness_parameters(args.mechanism, .5)),
                data_id=data_id, data_seeds=sim.encode(seeds), mask_seeds=sim.encode(mask_seeds),
                preprocess_center=sim.encode(transform.center), preprocess_std=sim.encode(transform.std),
                data_scale=transform.data_scale, evaluation_std=sim.encode(np.sqrt(np.diag(SIGMA))),
                reference_identity=sim.encode(ref_identity), reference_id=digest(ref_identity),
                regression_coordinates='raw_X_no_intercept', prior_var=5.,
                stop_protocol=sim.encode(dvai_stop_protocol(args.posterior_tol, k_max=args.K_max)),
                K_max=args.K_max, posterior_tol=args.posterior_tol, min_iter=dvai_config.min_iter, patience=dvai_config.patience,
                metric_protocol=METRICS, crps_protocol=CRPS_PROTOCOL,
                prediction_theta_mc=sim.PREDICTION_THETA_MC, acc_threshold=.5,
                timing_protocol=TIMING, timing_scope=TIMING,
                train_actual_missing_rate=float(masks[0].mean()), test_actual_missing_rate=float(masks[1].mean()))
    meta.update(scenario)
    return (xtr,xte,ytr,yte), masks, transform, meta


def validate_saved(row, meta, c=None):
    if str(row.get('rep')) != str(meta['rep']):
        raise ValueError('Saved repetition conflicts')
    for key in ('data_id','reference_id','reference_identity','preprocess_center','preprocess_std','stop_protocol'):
        if row.get(key) != meta[key]:
            raise ValueError(f'Saved result conflicts at {key}, rep={meta["rep"]}')
    for key in ('K_max','posterior_tol','min_iter','patience','master_seed'):
        if float(row[key]) != float(meta[key]):
            raise ValueError(f'Saved stopping/seed configuration conflicts at {key}')
    if c is not None:
        if row.get('config_id') != digest(c) or any(float(row[k]) != v for k,v in c.items()):
            raise ValueError('Saved candidate configuration conflicts')


def reference(store, data, meta, configs, args, counts):
    identity = f'reference/{meta["rep"]}'
    saved = active_record(store, identity)
    if saved:
        if saved.get('status') != SUCCESS:
            return None, saved
        mean, cov = sim.validate_full_vb(saved, json.loads(meta['reference_identity']))
        counts['reference_reuses'] += 1
        return (mean,cov), saved
    vc = deepcopy(configs['dvai_cholesky_gvb_config'])
    vc.seed = json.loads(meta['data_seeds'])['reference_vb']
    reference_timing = {}
    counts['reference_fits'] += 1
    print(f'Full-VB rep={meta["rep"]}', flush=True)
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with measure(args.device) as reference_timing:
                mean,cov = sim.fit_vb(data[0],data[2],vc,args.device)
        saved = dict(meta, row_type='reference', status=SUCCESS,
                     reference_mean=sim.encode(mean), reference_cov=sim.encode(cov),
                     reference_time_sec=reference_timing.get('seconds', 0.),
                     reference_timing_protocol=TIMING_PROTOCOL)
        result = (mean,cov)
    except Exception as exc:
        saved = dict(meta, row_type='failure', status='failed', failure_stage='reference_vb',
                     error=f'{type(exc).__name__}: {exc}', reference_time_sec=reference_timing.get('seconds', 0.),
                     reference_timing_protocol=TIMING_PROTOCOL)
        result = None
        print(saved['error'], flush=True)
    saved['numerical_warnings'] = '; '.join(str(w.message) for w in caught)
    store.commit(identity,[saved])
    return result,saved


def fit_candidate(store, c, data, masks, transform, meta, ref, ref_row, configs, args, counts):
    identity = task_id(meta['rep'],c)
    saved = active_record(store,identity)
    if saved:
        if saved.get('failure_stage') != 'data_preparation':
            validate_saved(saved,meta,c)
        counts['task_reuses' if successful(saved) else 'failed_skips'] += 1
        return saved
    common = dict(meta, **c, config_id=digest(c), candidate=candidate_id(c), method='dvai',
                  parameter_names=sim.encode([f'X{j+1}' for j in range(args.D)]),
                  regression_intercept=False, intercept_index=sim.encode(None),
                  covariance_encoding='full_matrix_json',
                  reference_mean=ref_row.get('reference_mean'), reference_cov=ref_row.get('reference_cov'),
                  reference_time_sec=ref_row.get('reference_time_sec'),
                  reference_timing_protocol=ref_row.get('reference_timing_protocol', 'legacy_unspecified'))
    if ref is None:
        row = dict(common,row_type='failure',status='failed',failure_stage='reference_vb',error=ref_row.get('error'),actual_k=0)
    else:
        local = deepcopy(configs)
        dc = local['diff_config']; dc.L=c['L']; dc.langevin_sigma=c['sigma_L']; dc.langevin_step_scale=ETA
        progress, rows, caught = {}, [], []
        row = dict(common)
        counts['dvai_fits'] += 1
        print(f'Run rep={meta["rep"]} {candidate_id(c)} full K_max=10', flush=True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                sim.run_trajectory(args, dict(name=candidate_id(c),J=c['J']), 'dvai', meta['rep'],args.mechanism,.5,
                                   *data, masks,ref,local,common,rows,progress,transform)
            final = rows[-1]
            keys = (*SCORES,'AUC_reason','test_crps_type','test_crps_reason','actual_k','attempted_k','stop_reason',
                    'posterior_mean','posterior_cov',
                    'first_stop_k','convergence_distance','consecutive','derived_seeds',
                    'train_time_sec','test_time_sec','total_time_sec')
            row = dict(common, **{k:final.get(k) for k in keys}, row_type='final',status=SUCCESS,
                       is_final=True, numerical_warnings='; '.join(str(w.message) for w in caught))
            if not successful(row): raise ValueError('Incomplete or nonfinite final result')
        except Exception as exc:
            row = dict(row,row_type='failure',status='failed',actual_k=progress.get('completed_k',0),
                       attempted_k=progress.get('attempted_k',0),failure_stage=progress.get('failure_stage'),
                       error=f'{type(exc).__name__}: {exc}',stop_reason='failed',
                       train_time_sec=progress.get('train_time_sec'),test_time_sec=progress.get('test_time_sec'),
                       numerical_warnings='; '.join(str(w.message) for w in caught))
        print({k:row.get(k) for k in ('rep','candidate','status',*SCORES,'actual_k','stop_reason','error')},flush=True)
    store.commit(identity,[row])
    return row


def summarize(store, candidates, reps):
    summaries = []
    print('candidate n_success/n overlap_mean overlap_mcse RMSE_mean MAE_mean CRPS_mean AUC_mean ACC_mean train_sec test_sec total_sec converged/max_iter',flush=True)
    for c in candidates:
        rows = [active_record(store,task_id(rep,c)) for rep in range(reps)]
        valid = [r for r in rows if successful(r)]
        mean = lambda key: float(np.mean([float(r[key]) for r in valid])) if valid else float('nan')
        mcse = float(np.std([float(r['overlap']) for r in valid],ddof=1)/np.sqrt(len(valid))) if len(valid)>1 else float('nan')
        summary = dict(candidate=c, n=len(valid), overlap=mean('overlap'),
                       rmse=mean('test_impute_rmse'), mae=mean('test_impute_mae'),
                       crps=mean('test_impute_crps'), auc=mean('AUC'), acc=mean('ACC'))
        summaries.append(summary)
        print(f'{candidate_id(c)} {len(valid)}/{reps} {summary["overlap"]:.8g} {mcse:.4g} {summary["rmse"]:.8g} {summary["mae"]:.8g} {summary["crps"]:.8g} '
              f'{summary["auc"]:.8g} {summary["acc"]:.8g} {mean("train_time_sec"):.3f} {mean("test_time_sec"):.3f} {mean("total_time_sec"):.3f} '
              f'{sum(r.get("stop_reason")=="converged" for r in valid)}/{sum(r.get("stop_reason")=="max_iter" for r in valid)}',flush=True)
        for rep,row in enumerate(rows):
            if not successful(row): print(f'  rep={rep}: {None if row is None else (row.get("failure_stage"),row.get("error"),row.get("status"))}',flush=True)
    return summaries


def select(summaries, reps):
    if ({candidate_id(s['candidate']) for s in summaries} != {candidate_id(c) for c in grid_candidates()}
            or len(summaries) != len(grid_candidates())
            or any(s['n'] != reps or not np.isfinite([s['overlap'],s['rmse'],s['mae'],s['crps'],s['auc'],s['acc']]).all() for s in summaries)):
        return None
    return sorted(summaries,key=lambda s:(-s['overlap'],s['rmse'],s['candidate']['L'],s['candidate']['sigma_L']))[0]['candidate']


def paired(store, left, right, reps):
    pairs = [(active_record(store,task_id(rep,left)),active_record(store,task_id(rep,right))) for rep in range(reps)]
    if not all(successful(a) and successful(b) for a,b in pairs):
        print(f'Paired comparison unavailable (missing successful fit or RMSE): {candidate_id(left)} - {candidate_id(right)}',flush=True)
        return
    report = dict(left=candidate_id(left),right=candidate_id(right),n=reps)
    for key in SCORES:
        delta=np.array([float(a[key])-float(b[key]) for a,b in pairs])
        report[key+'_difference']=float(delta.mean())
        report[key+'_mcse']=float(delta.std(ddof=1)/np.sqrt(reps)) if reps>1 else None
    report['RMSE_opposes_overlap']=report['overlap_difference']*report['test_impute_rmse_difference']>0
    report['close_within_2_mcse_descriptive']=abs(report['overlap_difference'])<=2*report['overlap_mcse']
    print(sim.encode(report),flush=True)


def copy_result(destination, source_row, identity, source_name):
    old = active_record(destination,identity)
    source_digest = digest({k:('' if v is None else str(v)) for k,v in source_row.items() if k not in ('trajectory','attempt','is_current')})
    if old:
        if old.get('source_digest') != source_digest:
            raise ValueError('Extracted result no longer matches its source')
        return
    row = {k:v for k,v in source_row.items() if k not in ('trajectory','attempt','is_current')}
    row.update(extracted_view=True,source_csv=source_name,source_trajectory=identity,source_digest=source_digest,
               independent_fit=False)
    destination.commit(identity,[row])


def run(args, configs, protocol, fingerprint, reps):
    counts=dict(dvai_fits=0,reference_fits=0,task_reuses=0,reference_reuses=0,failed_skips=0)
    invalid_references = []
    common=dict(protocol_revision=protocol['revision'], effective_config=sim.encode(protocol), source_sha256=protocol['source_sha256'],
                mechanism=args.mechanism, target_missing_rate=.5,
                stop_protocol=sim.encode(protocol['stop']),
                master_seed=args.master_seed, K_max=args.K_max, posterior_tol=args.posterior_tol,
                min_iter=dvai_config.min_iter, patience=dvai_config.patience,
                metric_protocol=METRICS, crps_protocol=CRPS_PROTOCOL,
                timing_protocol=TIMING, timing_scope=TIMING)
    with ExitStack() as stack:
        stores={key:stack.enter_context(ResultCSV(sensitivity_path(key,args.mechanism),fingerprint,protocol_details=protocol)) for key in ('sigma','l')}
        for rep in range(reps):
            existing = [active_record(stores['sigma'],task_id(rep,c)) for c in grid_candidates()]
            if all(r and r.get('failure_stage') == 'data_preparation' for r in existing):
                counts['failed_skips'] += len(existing)
                invalid_references.append(rep)
                continue
            try:
                data,masks,transform,meta=prepare(args,rep,configs)
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
                invalid_references.append(rep)
                print(f'rep={rep} data preparation failed: {error}',flush=True)
                for c in grid_candidates():
                    identity=task_id(rep,c)
                    if active_record(stores['sigma'],identity) is None:
                        stores['sigma'].commit(identity,[dict(common,**c,rep=rep,config_id=digest(c),
                            candidate=candidate_id(c),method='dvai',row_type='failure',status='failed',
                            failure_stage='data_preparation',error=error,actual_k=0,
                            is_final=True,stop_reason='failed')])
                continue
            meta.update(common)
            print(sim.encode(dict(rep=rep,train_missing=meta['train_actual_missing_rate'],test_missing=meta['test_actual_missing_rate'],
                                  initial_U_std=transform.to_network(transform.standardize(data[0],masks[0])).std(0))),flush=True)
            try:
                ref,ref_row=reference(stores['sigma'],data,meta,configs,args,counts)
            except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
                ref,ref_row=None,dict(error=f'{type(exc).__name__}: {exc}')
                print(f'rep={rep} reference validation failed: {ref_row["error"]}',flush=True)
            if ref is None: invalid_references.append(rep)
            for c in grid_candidates():
                fit_candidate(stores['sigma'],c,data,masks,transform,meta,ref,ref_row,configs,args,counts)
        summary=summarize(stores['sigma'],grid_candidates(),reps)
        chosen=None if invalid_references else select(summary,reps)
        if chosen is None:
            if invalid_references: print(f'Invalid/missing Full-VB references: {invalid_references}',flush=True)
            print('Selection and L view deferred: all 16 combinations require all planned reps with valid Overlap/RMSE/MAE/fair CRPS/AUC/ACC and timing; no automatic retries.',flush=True)
            print(sim.encode(counts),flush=True)
            return counts
        print('Grid-only selection by mean Overlap; not simultaneous six-metric or global parameter optimum: '+sim.encode(chosen),flush=True)
        print('Exact ties: mean Overlap descending, mean RMSE ascending, L ascending, sigma ascending.',flush=True)
        for c in grid_candidates():
            if c != chosen: paired(stores['sigma'],chosen,c,reps)
        for rep in range(reps):
            for L in L_VALUES:
                identity=task_id(rep,candidate(chosen['sigma_L'],L))
                copy_result(stores['l'],active_record(stores['sigma'],identity),identity,sensitivity_path('sigma',args.mechanism).name)
    print('Actual calls and reuse in this invocation: '+sim.encode(counts),flush=True)
    return counts


def preflight(fingerprint, protocol=None, *, mechanism='MAR'):
    conflicts = []
    for key in ('sigma','l'):
        path = sensitivity_path(key,mechanism)
        if not path.exists():
            continue
        from untils.result_io import read_result_rows
        rows = read_result_rows(path)
        if not rows or any(r.get('protocol_id') != fingerprint for r in rows):
            revisions = sorted({r.get('protocol_revision', '') for r in rows})
            tolerances = sorted({r.get('posterior_tol', '') for r in rows})
            timing = sorted({r.get('timing_protocol', '') for r in rows})
            detail = ''
            if protocol is not None:
                fields = set()
                def compare(old, new, prefix=''):
                    for name in set(old) | set(new):
                        before, after = old.get(name), new.get(name)
                        if isinstance(before, dict) and isinstance(after, dict):
                            compare(before, after, prefix + name + '.')
                        elif sim.encode(before) != sim.encode(after):
                            fields.add(prefix + name)
                for row in rows:
                    try:
                        old = json.loads(row.get('effective_config') or '{}')
                        if not isinstance(old, dict): raise ValueError('Expected object')
                        compare(old, json.loads(sim.encode(protocol)))
                    except (ValueError, TypeError):
                        fields.add('effective_config')
                detail = f'; conflicting_fields={sorted(fields)}'
            conflicts.append(f'{path}: revisions={revisions}, posterior_tol={tolerances}, '
                             f'timing_protocol={timing}, expected_timing={TIMING}, expected_protocol_id={fingerprint}' + detail)
            continue
        seen = set()
        for row in rows:
            if row.get('status') == SUCCESS and row.get('row_type') == 'final' and not successful(row):
                conflicts.append(f'{path}: invalid successful record {row.get("trajectory")}; six scores/timing/stopping required')
            if digest(json.loads(row['effective_config'])) != fingerprint:
                conflicts.append(f'{path}: protocol payload/hash mismatch')
            identity = row.get('trajectory')
            if not identity or identity in seen:
                conflicts.append(f'{path}: missing/duplicate trajectory {identity}')
            seen.add(identity)
    if conflicts:
        raise ValueError('Incompatible existing CSVs; no rows changed. Preserve/archive the listed files explicitly before running the new protocol:\n'
                         + '\n'.join(conflicts))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mechanism',choices=('MAR','MCAR'),default='MAR',help='Missingness mechanism at a fixed 50%% rate (default MAR)')
    parser.add_argument('--reps',type=int,choices=(5,20),default=5,help='Batch repetitions (default 5); matching completed reps are reused')
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--dry-run',action='store_true',help='Validate configuration and existing protocols without fitting or writing')
    options=parser.parse_args(argv)
    configure_single_thread()
    device=('cuda' if torch.cuda.is_available() else 'cpu') if options.device=='auto' else options.device
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    args,configs,protocol,fingerprint=study(device,options.mechanism)
    print(sim.encode(dict(design=design_parameters(),missingness=protocol['missingness'],stop=protocol['stop'],
                          grids=protocol['grids'],metrics=SCORES,prediction=protocol['prediction'],timing_protocol=TIMING,
                          device=protocol['device_name'],rep_range=[0,options.reps-1],
                          max_distinct_DVAI_fits=len(grid_candidates())*options.reps,Full_VB_fits=options.reps,
                          outputs={k:str(sensitivity_path(k,args.mechanism)) for k in ('sigma','l')},protocol_id=fingerprint)),flush=True)
    preflight(fingerprint, protocol, mechanism=args.mechanism)
    if options.dry_run:
        print('Dry run passed: 16 sigma/L combinations; fixed J=20; K_max=10; final-only imputation and prediction planned; no fitting, prediction or files written.',flush=True)
        return
    set_global_seed(args.master_seed)
    run(args,configs,protocol,fingerprint,options.reps)


if __name__=='__main__':
    main()
