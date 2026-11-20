from pathlib import Path
import argparse
import hashlib
import io
import json
import sys
import time
from copy import deepcopy
from dataclasses import asdict, replace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import norm, gaussian_kde
from untils.metrics import classification_metrics
from sklearn.metrics import log_loss
from configs import dvai_cholesky_gvb_config
from generate_miss.fixed_design import sample_x, sample_y, THETA, design_parameters
from untils.seed_device import derive_seed, configure_single_thread
from untils.cholesky_gvb import CholeskyGaussianVB, CholeskyGVBConfig
from untils.logistic_regression import BayesianLogisticRegression as BLR, BayesianLogisticRegressionMC as BLRMC
from untils.mcmc_mala import run_mala_chains
from untils.time import measure, TIMING_PROTOCOL

OUTPUT = ROOT / 'results/testoutput/cholesky_gvb.csv'
CHECK_RESULTS = {}

def save(name, value):
    CHECK_RESULTS[name] = value

def task(x, y):
    return BLR(torch.as_tensor(x, dtype=torch.float64),
               torch.as_tensor(y, dtype=torch.float64), prior_var=5.)

def checks():
    rng=np.random.default_rng(81); x=rng.normal(size=(17,3)); y=rng.binomial(1,.5,17)
    t=task(x,y); th=torch.tensor(rng.normal(size=(7,3)),requires_grad=True)
    a=torch.autograd.grad(t.log_joint(th).sum(),th)[0].numpy()
    b=(y[:,None]-expit(x@th.detach().numpy().T)).T@x-th.detach().numpy()/5
    errors={'analytic_autograd':float(np.max(abs(a-b)))}
    cfg=CholeskyGVBConfig(max_iter=1,seed=67,lr=.003,step_adaptive=1)
    vb=CholeskyGaussianVB(3,cfg).double()
    C=np.array([[-.8,0,0],[.2,.7,0],[.1,-.3,-.5]]); mu=np.array([.3,-.2,.1])
    with torch.no_grad(): vb.mu.copy_(torch.tensor(mu)); vb.C.copy_(torch.tensor(C))
    eps=rng.normal(size=(13,3)); theta=mu+eps@C.T
    score=(y[:,None]-expit(x@theta.T)).T@x-theta/5+(theta-mu)@np.linalg.inv(C@C.T)
    gc=score.T@eps/len(eps)
    independent=np.r_[score.mean(0),[gc[i,j] for j in range(3) for i in range(j,3)]]
    g,lb=vb.estimate(t,torch.tensor(eps))
    logjoint=(y[:,None]*(x@theta.T)-np.logaddexp(0,x@theta.T)).sum(0)-.5*(theta**2).sum(1)/5-1.5*np.log(2*np.pi*5)
    logq=-1.5*np.log(2*np.pi)-np.log(abs(np.linalg.det(C)))-.5*(eps**2).sum(1)
    errors['signed_score']=float(np.max(abs(g.numpy()-independent)))
    errors['elbo']=float(abs(lb.item()-np.mean(logjoint-logq)))
    mt=BLRMC(t.X[None],t.y,prior_var=5.)
    errors['J1_value']=float((mt.log_joint(th)-t.log_joint(th)).detach().abs().max())
    errors['J1_gradient']=float((torch.autograd.grad(mt.log_joint(th).sum(),th)[0]-torch.tensor(a)).abs().max())
    class Gaussian:
        def log_joint(self,theta):
            d=theta-torch.tensor(mu)
            return -.5*torch.einsum('si,ij,sj->s',d,torch.tensor(np.linalg.inv(C@C.T)),d)
    errors['gaussian_zero_score']=float(vb.estimate(Gaussian(),torch.tensor(eps))[0].abs().max())
    vb=CholeskyGaussianVB(3,cfg).double(); gen=torch.Generator().manual_seed(67)
    gradients=[]
    for _ in range(2):
        e=torch.randn(20,3,generator=gen,dtype=torch.float64).numpy(); theta=.1*e
        vs=(y[:,None]-expit(x@theta.T)).T@x-theta/5+e/.1
        G=vs.T@e/20
        gradients.append(np.r_[vs.mean(0),[G[i,j] for j in range(3) for i in range(j,3)]])
    m=.9*gradients[0]+.1*gradients[1]; v=.9*gradients[0]**2+.1*gradients[1]**2
    expected=.003*m/np.maximum(np.sqrt(v),1e-12)
    out=vb.fit(t); actual=np.r_[out['mu'],[(vb.C.detach().numpy()-.1*np.eye(3))[i,j] for j in range(3) for i in range(j,3)]]
    errors['one_step']=float(np.max(abs(actual-expected)))
    class SequenceVB(CholeskyGaussianVB):
        def __init__(self,values,**kw): super().__init__(1,CholeskyGVBConfig(**kw)); self.values=iter(values)
        def final_elbo(self, task): return {}
        def estimate(self,task,epsilon): return torch.zeros(2,dtype=self.mu.dtype),torch.tensor(next(self.values),dtype=self.mu.dtype)
    seq=SequenceVB([0,3,3,2,1,0],window=2,patience=2,max_iter=5).double().fit(None)
    assert seq['actual_steps']==4 and seq['patience_history']==[0,0,1,2]
    tie=SequenceVB([0]+[3]*6,window=2,patience=1,max_iter=6,step_adaptive=2).double().fit(None)
    assert tie['actual_steps']==6 and tie['best_step']==5 and tie['patience_history']==[0]*6
    assert np.allclose(tie['learning_rate_history'],.002*np.minimum(1,2/np.arange(1,7)))
    assert tie['denominator_floor_activations']==12
    class RestoreVB(SequenceVB):
        def estimate(self,task,epsilon):
            return torch.ones(2,dtype=self.mu.dtype),torch.tensor(next(self.values),dtype=self.mu.dtype)
    restored=RestoreVB([0,3,3,2],window=2,patience=1,max_iter=3,restore_best=True).double().fit(None)
    assert restored['actual_steps']==3 and restored['returned_completed_updates']==1
    assert restored['best_step']==1 and abs(restored['mu'][0]-.002)<1e-12
    warm=CholeskyGaussianVB(3,cfg).double(); warm.set_init_from_mu_cov(mu,C@C.T)
    errors['warm_mean']=float(np.max(abs(warm.mu.detach().numpy()-mu)))
    errors['warm_cov']=float(np.max(abs((warm.C@warm.C.T).detach().numpy()-C@C.T)))
    warm2=CholeskyGaussianVB(3,cfg).double(); warm2.load_state_dict(warm.state_dict())
    errors['checkpoint']=float((warm2.C-warm.C).detach().abs().max())
    try: warm2.load_state_dict({'L_unconstrained':torch.eye(3)},strict=False)
    except ValueError: pass
    else: raise AssertionError('legacy checkpoint accepted')
    for name,value in errors.items(): assert value<1e-6,(name,value)
    save('deterministic_checks.json',dict(errors=errors,stop_checks='passed',legacy_checkpoint='rejected',
        checkpoint_scope='parameters only; fit restarts moments and stopping window',
        dtype='float64',device='cpu'))
    print('Deterministic checks PASSED',errors,flush=True)


def interface_checks():
    checks_out={}
    for field,value in [('check_every',10),('min_improve',1e-4),('monitor_seed',42)]:
        cfg=deepcopy(dvai_cholesky_gvb_config); setattr(cfg,field,value)
        try: CholeskyGaussianVB(1,cfg).fit(task([[1.],[2.]],[0.,1.]))
        except ValueError as exc: checks_out[field]=str(exc)
        else: raise AssertionError(field)
    v=CholeskyGaussianVB(2,CholeskyGVBConfig(max_iter=2)).double()
    t=task([[1.,2.],[-1.,1.]],[0.,1.])
    for bad in [0.,float('nan')]:
        with torch.no_grad(): v.C[0,0]=bad
        try: v.estimate(t,torch.ones(2,2))
        except FloatingPointError: pass
        else: raise AssertionError('invalid factor accepted')
    v.set_init_from_mu_cov(np.zeros(2),np.eye(2))
    ref=t.log_joint(torch.ones(1,2)).detach().clone()
    v.fit(t); v.fit(t)
    assert torch.equal(ref,t.log_joint(torch.ones(1,2)))
    checks_out['fixed_prior_repeated_fit']='unchanged'
    buf=io.BytesIO(); torch.save(v.state_dict(),buf); buf.seek(0)
    w=CholeskyGaussianVB(2,v.cfg).double(); w.load_state_dict(torch.load(buf,weights_only=True))
    assert torch.equal(v.mu,w.mu) and torch.equal(v.C,w.C)
    checks_out['parameter_checkpoint_disk_format']='exact roundtrip; no optimizer resume claimed'
    torch.set_default_dtype(torch.float32)
    t=task([[0.,0.]],[0.]); val=t.log_joint(torch.zeros(1,2,dtype=torch.float64)).item()
    assert abs(val-(-np.log(2)-np.log(2*np.pi*5)))<1e-12

    torch.set_default_dtype(torch.float64)
    CHECK_RESULTS["interfaces"] = checks_out

def encode(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False,
                      default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())


def diagnose(samples):
    import arviz as az
    import xarray as xr
    ds = xr.Dataset({'theta': (('chain', 'draw', 'coefficient'), samples)})
    values = lambda fn, **kw: fn(ds, **kw)['theta'].values
    sd = samples.reshape(-1, samples.shape[-1]).std(0, ddof=1)
    return dict(rhat=values(az.rhat, method='rank'),
                ess_bulk=values(az.ess, method='bulk'),
                ess_tail=values(az.ess, method='tail'),
                mcse_mean=values(az.mcse, method='mean'), sd=sd)


def prediction(draws, x, y):
    independent = draws.ndim == 2
    if independent:
        draws = draws[None]
    means, errors = [], []
    for start in range(0, len(x), 20):
        p = expit(draws @ x[start:start+20].T)
        means.extend(p.mean((0, 1)))
        if independent:
            errors.extend(p[0].std(0, ddof=1)/np.sqrt(draws.shape[1]))
        else:
            import arviz as az
            import xarray as xr
            ds = xr.Dataset({'p': (('chain', 'draw', 'observation'), p)})
            errors.extend(az.mcse(ds, method='mean')['p'].values)
    p = np.asarray(means)
    return dict({k:v for k,v in classification_metrics(y, p).items() if k != "p_test"},
                log_loss=log_loss(y, p, labels=[0,1]), max_probability_mcse=max(errors))


def run_rep(args, rep):
    seeds = {p: derive_seed(args.master_seed, 'formal', rep, p) for p in ['data', 'response']}
    x = sample_x(1000, np.random.default_rng(seeds['data']))
    y = sample_y(x, np.random.default_rng(seeds['response']))
    target = task(x[:700], y[:700])
    a = x[:700]
    opt = minimize(lambda b: np.sum(np.logaddexp(0, a@b)-y[:700]*(a@b))+b@b/10,
                   np.zeros(10), jac=lambda b: a.T@(expit(a@b)-y[:700])+b/5,
                   method='BFGS')
    center = opt.x
    p = expit(a@center)
    h = a.T@((p*(1-p))[:, None]*a)+np.eye(10)/5
    factor = np.linalg.cholesky(np.linalg.solve(h, np.eye(10)))
    class Coordinates:
        X = target.X
        D = 10
        def log_joint(self, z):
            return target.log_joint(torch.as_tensor(center)+z@torch.as_tensor(factor).T)+np.linalg.slogdet(factor)[1]
    initial_seed = derive_seed(args.master_seed, 'gvb_examine', rep, 'initial')
    initials = np.random.default_rng(initial_seed).normal(size=(4, 10))*2
    chain_seeds = [derive_seed(args.master_seed, 'gvb_examine', rep, 'chain', i) for i in range(4)]
    started = time.perf_counter()
    z, chain_info = run_mala_chains(Coordinates(), initials, chain_seeds,
        num_mcmc=args.warmup+args.draws, burnin=args.warmup/(args.warmup+args.draws),
        thin=1, step_init=.3, target_acc=.57, adapt_until=args.warmup)
    mcmc_seconds = time.perf_counter()-started
    samples = center+z@factor.T
    diag = diagnose(samples)
    ratio = diag['mcse_mean']/diag['sd']
    passed = bool(np.all(diag['rhat'] < 1.01) and np.all(diag['ess_bulk'] >= 400)
                  and np.all(diag['ess_tail'] >= 400))
    flat = samples.reshape(-1, 10)
    mm, mc = flat.mean(0), np.cov(flat, rowvar=False)
    ms = np.sqrt(mc.diagonal())
    ml, mh = np.quantile(flat, [.025, .975], axis=0)
    mp = prediction(samples, x[700:], y[700:])
    oracle_p = expit(x[700:]@THETA)
    oracle = dict({k:v for k,v in classification_metrics(y[700:], oracle_p).items() if k != "p_test"},
                  log_loss=log_loss(y[700:], oracle_p, labels=[0,1]),
                  expected_ACC=np.maximum(oracle_p, 1-oracle_p).mean())
    base = dict(schema_version=2, source='fresh_fit',
        generation_mechanism='single_gaussian', design_parameters=encode(design_parameters()), rep=rep, master_seed=args.master_seed,
        data_seed=seeds['data'], response_seed=seeds['response'], n_train=700, n_test=300,
        data_sha256=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest(),
        prior_var=5., threshold=.5, dtype='float64', sampler='MALA', chains=4,
        warmup=args.warmup, retained_per_chain=args.draws, thin=1,
        mcmc_diagnostic_pass=passed, mcmc_precision_pass=bool(np.all(ratio <= .02)),
        mcmc_seconds=mcmc_seconds, mcmc_coefficient_rmse=np.sqrt(np.mean((mm-THETA)**2)),
        chain_info=encode(chain_info), coordinate_center=encode(center), coordinate_factor=encode(factor),
        initial_seed=initial_seed, coordinate_initials=encode(initials),
        mcmc_covariance=encode(mc), train_positive=y[:700].mean(), test_positive=y[700:].mean(),
        **{f'mcmc_{k}':v for k,v in mp.items()}, **{f'oracle_{k}':v for k,v in oracle.items()})
    rows = []
    for seed in args.vb_seeds:
        vb_seed = seed if rep == 0 else derive_seed(seed, 'gvb_examine', rep, 'vb')
        cfg = replace(dvai_cholesky_gvb_config, seed=vb_seed)
        vb = CholeskyGaussianVB(10, cfg).double()
        with measure(vb.mu.device) as measured:
            r = vb.fit(target)
        elapsed = measured['seconds']
        sd = np.sqrt(r['cov'].diagonal())
        low, high = r['mu']-norm.ppf(.975)*sd, r['mu']+norm.ppf(.975)*sd
        pred_seed = derive_seed(seed, 'gvb_examine', rep, 'prediction')
        draws = np.random.default_rng(pred_seed).multivariate_normal(r['mu'], r['cov'], args.prediction_draws)
        pred = prediction(draws, x[700:], y[700:])
        run = dict(base, vb_seed=seed, vb_actual_seed=vb_seed, vb_config=encode(asdict(cfg)),
            vb_seconds=elapsed, vb_timing_protocol=TIMING_PROTOCOL, vb_steps=r['actual_steps'], vb_stop=r['inner_stop_reason'],
            vb_final_elbo=r['final_elbo'], vb_final_elbo_mcse=r['final_elbo_mcse'],
            vb_coefficient_rmse=np.sqrt(np.mean((r['mu']-THETA)**2)), vb_covariance=encode(r['cov']),
            covariance_relative_frobenius=np.linalg.norm(r['cov']-mc)/np.linalg.norm(mc),
            vb_prediction_draws=args.prediction_draws, vb_prediction_seed=pred_seed,
            **{f'vb_{k}':v for k,v in pred.items()})
        for j in range(10):
            grid = np.linspace(min(ml[j], low[j], THETA[j])-ms[j], max(mh[j], high[j], THETA[j])+ms[j], 200)
            rows.append(dict(run, coefficient=j+1, theta_true=THETA[j],
                vb_mean=r['mu'][j], vb_sd=sd[j], vb_lower=low[j], vb_upper=high[j],
                mcmc_mean=mm[j], mcmc_sd=ms[j], mcmc_lower=ml[j], mcmc_upper=mh[j],
                standardized_mean_difference=(r['mu'][j]-mm[j])/ms[j], sd_ratio=sd[j]/ms[j],
                vb_covers_truth=bool(low[j]<=THETA[j]<=high[j]), mcmc_covers_truth=bool(ml[j]<=THETA[j]<=mh[j]),
                rhat=diag['rhat'][j], ess_bulk=diag['ess_bulk'][j], ess_tail=diag['ess_tail'][j],
                mcse_mean=diag['mcse_mean'][j], mcse_over_sd=ratio[j],
                density_x=encode(grid), mcmc_density=encode(gaussian_kde(flat[:, j])(grid)),
                vb_density=encode(norm.pdf(grid, r['mu'][j], sd[j]))))
        print(f'rep={rep} vb_seed={seed} steps={r["actual_steps"]} MCMC_pass={passed}', flush=True)
    return rows


def write_csv(rows, path):
    frame = pd.DataFrame(rows)
    if frame.duplicated(['rep','vb_seed','coefficient']).any():
        raise ValueError('Duplicate coefficient records')
    for _, idx in frame.groupby(['vb_seed','coefficient']).groups.items():
        group=frame.loc[idx]; n=group.rep.nunique()
        frame.loc[idx,'data_repetitions']=n
        if n>1:
            for col in ['vb_covers_truth','mcmc_covers_truth','vb_mean','vb_sd','mcmc_mean','mcmc_sd',
                        'vb_AUC','vb_ACC','vb_Brier','vb_log_loss','mcmc_AUC','mcmc_ACC','mcmc_Brier','mcmc_log_loss']:
                values=group[col].astype(float)
                for suffix,value in [('mean',values.mean()),('sd',values.std(ddof=1)),('mcse',values.std(ddof=1)/np.sqrt(n))]:
                    frame.loc[idx,f'across_data_{col}_{suffix}']=value
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.csv.tmp')
    frame.to_csv(temp,index=False)
    temp.replace(path)
    print(f'Saved {path}: {len(frame)} coefficient rows',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,choices=[1,10],default=1)
    parser.add_argument('--vb-seeds',type=int,nargs='+',default=[101])
    parser.add_argument('--master-seed',type=int,default=2026)
    parser.add_argument('--warmup',type=int,default=10000)
    parser.add_argument('--draws',type=int,default=20000)
    parser.add_argument('--prediction-draws',type=int,default=40000)
    parser.add_argument('--checks-only',action='store_true')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if min(args.warmup,args.draws,args.prediction_draws)<2 or len(set(args.vb_seeds))!=len(args.vb_seeds):
        parser.error('Positive budgets >=2 and distinct VB seeds required')
    if args.checks_only:
        configure_single_thread();torch.set_default_dtype(torch.float64)
        checks();interface_checks();print('Deterministic/interface checks passed');return
    from untils.result_io import result_directory
    args.output = args.output.resolve() if args.output.is_absolute() else (ROOT / args.output).resolve()
    if args.output.parent != result_directory('testoutput').resolve():
        parser.error('GVB check output must be a file directly in results/testoutput')
    if args.output.exists() and not args.overwrite:
        parser.error('Output exists; choose --output or explicitly --overwrite')
    configure_single_thread();torch.set_default_dtype(torch.float64)
    checks();interface_checks()
    rows=[]
    for rep in range(args.repeats):
        rows.extend(run_rep(args,rep));write_csv(rows,args.output)
    print('MCMC diagnostic failures remain flagged; no automatic extension or method selection.')


if __name__=='__main__':
    main()
