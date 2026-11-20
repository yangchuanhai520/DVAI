import numpy as np


def validate_observed(x, mask, *, training=False, feature_names=None, intercept_index=None,
                      context='', nan_missing=False, mechanism=None, target_rate=None):
    x, mask = np.asarray(x, dtype=float), np.asarray(mask)
    prefix = f'{context}: ' if context else ''
    if x.ndim != 2 or mask.ndim != 2 or x.shape != mask.shape or min(x.shape) == 0:
        raise ValueError(prefix + "Data and mask must be nonempty matching two-dimensional arrays")
    if mask.dtype != np.bool_:
        raise TypeError(prefix + "mask must have boolean dtype")
    names = list(feature_names) if feature_names is not None else [f'X{j+1}' for j in range(x.shape[1])]
    if len(names) != x.shape[1]:
        raise ValueError(prefix + 'Feature names must match matrix columns')
    columns = np.ones(x.shape[1], dtype=bool)
    if intercept_index is not None:
        if not isinstance(intercept_index, (int, np.integer)) or not 0 <= intercept_index < x.shape[1]:
            raise ValueError(prefix + 'Invalid intercept column index')
        if mask[:, intercept_index].any():
            raise ValueError(prefix + 'Regression intercept must remain observed')
        columns[intercept_index] = False
    describe = lambda indices: ', '.join(f'{names[j]}[column={j+1}]' for j in indices)
    if nan_missing and not np.array_equal(np.isnan(x), mask):
        bad = np.flatnonzero((np.isnan(x) != mask).any(0))
        raise ValueError(prefix + 'NaN/missing-mask mismatch: ' + describe(bad))
    bad = np.flatnonzero((~np.isfinite(x) & ~mask).any(0))
    if bad.size:
        raise ValueError(prefix + 'Non-finite observed values: ' + describe(bad))
    if training:
        bad = np.flatnonzero(columns & ((~mask).sum(0) == 0))
        if bad.size:
            raise ValueError(prefix + 'No training observations (observed_count=0): ' + describe(bad))
    if mechanism == 'MAR':
        from generate_miss.data_miss import mar_target_rate
        mar_target_rate(int(columns.sum()), target_rate)
        driver = np.flatnonzero(columns)[0]
        if mask[:, driver].any():
            raise ValueError(prefix + 'MAR driver must remain observed: ' + describe([driver]))
    return x, mask


class ObservedTransform:
    def __init__(self, train, mask, data_scale=2., *, context='', mechanism=None, target_rate=None):
        train, mask = validate_observed(train, mask, training=True, context=context,
                                        mechanism=mechanism, target_rate=target_rate)
        observed = np.where(mask, np.nan, train)
        self.center = np.nanmean(observed, axis=0)
        self.std = np.nanstd(observed, axis=0)
        self.degenerate = np.flatnonzero(self.std < 1e-8).tolist()
        self.std[self.std < 1e-8] = 1.
        self.data_scale = float(data_scale)
        if not np.isfinite(self.data_scale) or self.data_scale <= 0:
            raise ValueError('data_scale must be positive and finite')
        self.scale = self.data_scale * self.std

    def _validate(self, x, mask):
        x, mask = validate_observed(x, mask)
        if x.shape[1] != len(self.center):
            raise ValueError("Feature count differs from fitted transform")
        return x, mask

    def standardize(self, x, mask):
        x, mask = self._validate(x, mask)
        return np.where(mask, 0., (x - self.center) / self.std).astype(np.float32)

    def to_network(self, z):
        return np.asarray(z) / self.data_scale

    def from_network(self, u):
        return np.asarray(u, dtype=float) * self.data_scale

    def unstandardize(self, samples, original, mask):
        original, mask = self._validate(original, mask)
        if np.asarray(samples).ndim not in (2, 3) or np.asarray(samples).shape[-2:] != original.shape:
            raise ValueError("Completed samples must match original data shape")
        raw = np.asarray(samples, dtype=float) * self.std + self.center
        if raw.ndim == 3:
            raw[:, ~mask] = original[~mask]
        else:
            raw[~mask] = original[~mask]
        if not np.isfinite(raw).all():
            raise FloatingPointError('Non-finite completed covariates')
        return raw

    def forward(self, x, mask):
        x, mask = self._validate(x, mask)
        return np.where(mask, 0., (x - self.center) / self.scale).astype(np.float32)

    def inverse(self, samples, original, mask):
        original, mask = self._validate(original, mask)
        if np.asarray(samples).ndim not in (2, 3) or np.asarray(samples).shape[-2:] != original.shape:
            raise ValueError("Completed samples must match original data shape")
        raw = np.asarray(samples, dtype=float) * self.scale + self.center
        if raw.ndim == 3:
            raw[:, ~mask] = original[~mask]
        else:
            raw[~mask] = original[~mask]
        if not np.isfinite(raw).all():
            raise FloatingPointError("Non-finite completed covariates")
        return raw
