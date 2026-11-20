from untils.metrics import classification_metrics

from untils.time import timed_method, phase, in_phase, measure
import time
import numpy as np
import torch
from models.diffusion_utils import impute_mask, validate_budget
from models.diffusion_imputer import ConditionalDiffusionImputer
from configs import DiffusionConfig
from untils.inference import fit_vb, posterior_predict_proba, estimate_fields
from untils.data_process import ObservedTransform
from untils.seed_device import derive_seed


def _train(completed,cfg,device,init_seed,training_seed):
    shared = DiffusionConfig()
    for name in ('hidden', 'lr', 'epochs', 'batch_size', 'sample_steps'):
        setattr(shared, name, getattr(cfg, name))
    shared.init_seed, shared.train_seed = init_seed, training_seed
    imputer = ConditionalDiffusionImputer(completed.shape[1], shared, torch.device(device))
    info = imputer.fit(completed)
    return imputer.model, info


def _samples(model,x,mask,cfg,device,seed,intercept_index=None):
    if intercept_index is not None:
        columns = np.arange(x.shape[1]) != intercept_index
        draws = _samples(model, x[:, columns], mask[:, columns], cfg, device, seed)
        samples = np.broadcast_to(x, (cfg.num_imputations, *x.shape)).copy()
        samples[:, :, columns] = draws
        return samples
    torch.manual_seed(seed)
    x_t = torch.as_tensor(x,dtype=torch.float32,device=device)
    m_t = torch.as_tensor(mask,dtype=torch.bool,device=device)
    samples = np.stack([impute_mask(model,x_t,m_t,len(x),x.shape[1],cfg.sample_steps,device,
                                   refine_steps=cfg.refine_steps).cpu().numpy()
                        for _ in range(cfg.num_imputations)])
    if not np.isfinite(samples).all(): raise FloatingPointError("Non-finite DiffPuter samples")
    samples[:,~mask] = x[~mask]
    return samples


@timed_method
def run_diffputer_baseline(X_obs_train,y_train,X_obs_test,y_test,train_mask_missing,
        test_mask_missing,prior_var,vb_cfg,diffputer_cfg,device,theta_mc_samples=50,
        regression_transform=None,raw_observed=None,intercept_index=None):
    progress = dict(K_max=diffputer_cfg.K, k=0, actual_k=0, attempted_k=0, failure_stage="budget_validation")
    try:
        for name in ("K", "epochs", "num_imputations", "batch_size", "refine_steps"):
            validate_budget(getattr(diffputer_cfg, name), name)
        validate_budget(diffputer_cfg.sample_steps, "sample_steps", 2)
        progress["failure_stage"] = "preprocessing"
        masks = [np.asarray(train_mask_missing,bool),in_phase("test", lambda: np.asarray(test_mask_missing,bool))]
        if regression_transform is None:
            transform = ObservedTransform(X_obs_train,masks[0],2.)
            original = [np.asarray(X_obs_train),in_phase("test", lambda: np.asarray(X_obs_test))]
            data = [in_phase("test" if i else "train", lambda: transform.forward(x,m)) for i,(x,m) in enumerate(zip(original,masks))]
            restore = transform.inverse
        else:
            original = raw_observed
            data = [np.asarray(X_obs_train)/2.,in_phase("test", lambda: np.asarray(X_obs_test)/2.)]
            restore = lambda x,raw,mask: regression_transform.unstandardize(x*2.,raw,mask)
        data = [in_phase("test" if i else "train", lambda: np.where(m,0.,x).astype(np.float32)) for i,(x,m) in enumerate(zip(data,masks))]
        columns = np.arange(data[0].shape[1]) != intercept_index if intercept_index is not None else np.ones(data[0].shape[1], dtype=bool)
        current = data[0].copy()
        master = int(getattr(diffputer_cfg,"seed",2026))
        history = []
        training_sec = imputation_sec = 0.
        for k in range(1,diffputer_cfg.K+1):
            progress.update(attempted_k=k, failure_stage="diffusion_training")
            seeds = {p:derive_seed(master,"formal",0,p,k) for p in ["network_init","training","sampling_train","sampling_test"]}
            with measure(device) as measured:
                model,info = _train(current[:, columns],diffputer_cfg,device,seeds["network_init"],seeds["training"])
            elapsed_training = measured['seconds']
            training_sec += elapsed_training
            t = time.perf_counter()
            progress["failure_stage"] = "train_sampling"
            samples = _samples(model,data[0],masks[0],diffputer_cfg,device,seeds["sampling_train"],intercept_index)
            current = samples.mean(0)
            current[~masks[0]] = data[0][~masks[0]]
            imputation_sec += time.perf_counter()-t
            with phase("evaluation"):
                history.append(dict(k=k,seeds=seeds,training=info))
            progress.update(k=k, actual_k=k)
        t = time.perf_counter()
        progress["failure_stage"] = "test_sampling"
        with phase("test"):
            test_samples = _samples(model,data[1],masks[1],diffputer_cfg,device,seeds["sampling_test"],intercept_index)
        progress["failure_stage"] = "regression_transform"
        train = restore(current,original[0],masks[0])
        with phase("test"):
            test = restore(test_samples.mean(0),original[1],masks[1])
        imputation_sec += time.perf_counter()-t
        with measure(device) as measured:
            progress["failure_stage"] = "vb_fit"
            mu,cov,info = fit_vb(train,y_train,prior_var,vb_cfg,device,True)
        inference_sec = measured['seconds']
        t = time.perf_counter()
        progress["failure_stage"] = "prediction"
        with phase("test"):
            p_test = posterior_predict_proba(test, mu, cov, device, theta_mc_samples, int(getattr(vb_cfg, 'prediction_seed', vb_cfg.seed)))
        prediction_sec = time.perf_counter()-t
        metrics = in_phase("evaluation", lambda: classification_metrics(y_test, p_test))
        with phase("evaluation"):
            return dict(metrics,**estimate_fields(mu,cov,True),method="diffputer",mu=mu,cov=cov,
                        X_train_imp=train,X_test_imp=test,gvb_fits=1,regression_input_shape=list(train.shape),
                        training_sec=training_sec,imputation_sec=imputation_sec,inference_sec=inference_sec,
                        prediction_sec=prediction_sec,
                        elbo_last=info["elbo_last"],outer_history=history,K_max=diffputer_cfg.K,
                        k=diffputer_cfg.K,actual_k=diffputer_cfg.K,attempted_k=diffputer_cfg.K,
                        diffputer_num_vb_fits=1,diffputer_uses_mc_vb=False,stop_reason="fixed_outer_budget",
                        status="completed",supports_oos=True)
    except Exception as exc:
        exc.baseline_progress = progress.copy()
        raise
