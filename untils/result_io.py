import os
import csv
import pandas as pd



CSV_FIELD_LIMIT = 100 * 1024 * 1024


def configure_csv_reader():
    csv.field_size_limit(max(csv.field_size_limit(), CSV_FIELD_LIMIT))


def read_result_rows(path):
    from pathlib import Path
    configure_csv_reader()
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def source_version(entry=None):
    import hashlib
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    files = [root / 'configs.py']
    entries = [entry] if entry else ['simulation_main.py','support2_main.py']
    files += [root / name for name in entries]
    if entry and entry == 'scripts/test/parameter_test.py':
        files += [root/'simulation_main.py']
    for folder in ['models','baseline','generate_miss','untils'] + ([] if entry else ['scripts/test']):
        files += sorted((root / folder).glob('*.py'))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    try:
        result = subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True)
        head = result.stdout.strip() or 'unborn'
    except FileNotFoundError:
        head = 'git_unavailable'
    return {'git_head': head, 'source_sha256': digest.hexdigest()}


def runtime_identity(device):
    import importlib.metadata
    import platform
    import subprocess
    import torch
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if str(device) == 'auto' else device)
    versions = {n: importlib.metadata.version(n) for n in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'threadpoolctl')}
    versions.update(torch=str(torch.__version__), python=platform.python_version(),
                    platform=platform.system(), machine=platform.machine(),
                    cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(), device=device.type,
                    torch_distributions=sorted(d.version for d in importlib.metadata.distributions()
                                               if d.metadata.get('Name', '').lower() == 'torch'))
    if device.type == 'cuda':
        versions.update(gpu=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)))
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=10, check=True)
            versions['driver'] = sorted(set(result.stdout.splitlines()))
        except (OSError, subprocess.SubprocessError):
            versions['driver'] = 'unavailable'
    else:
        versions['processor'] = platform.processor()
    return versions


class ResultCSV:
    def __init__(self, path, fingerprint, protocol_details=None, *, validate_existing=None):
        from pathlib import Path
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(self.path.suffix + '.lock')
        self.fingerprint = fingerprint
        self.protocol_details = protocol_details
        self.validate_existing = validate_existing
        self.rows = []

    def __enter__(self):
        self.fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(self.fd, str(os.getpid()).encode())
        try:
            if self.path.exists():
                self.rows = read_result_rows(self.path)
                if self.validate_existing is not None:
                    self.validate_existing(self.rows)
                elif any(r.get('protocol_id') != self.fingerprint for r in self.rows):
                    import json
                    conflicts = []
                    if self.protocol_details is not None and self.rows:
                        try:
                            old = json.loads(self.rows[0].get('effective_config', '{}'))
                        except (ValueError, TypeError):
                            old = {}
                        def compare(before, after, prefix=''):
                            for key in sorted(set(before) | set(after)):
                                if key in {'out_dir','rep_start','reps','retry_failed'}: continue
                                name = prefix+key
                                if isinstance(before.get(key), dict) and isinstance(after.get(key), dict):
                                    compare(before[key], after[key], name+'.')
                                elif before.get(key) != after.get(key): conflicts.append(name)
                        compare(old, self.protocol_details)
                    detail = ', '.join(conflicts) if conflicts else 'source/dependencies/protocol_id'
                    raise ValueError(f'Incompatible existing CSV {self.path}: conflict fields: {detail}. No rows changed; resolve the existing-file conflict explicitly before rerunning.')
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def current_rows(self, trajectory):
        rows = [r for r in self.rows if r.get('trajectory') == trajectory
                and r.get('row_type') != 'failure_history']
        latest = max((int(r.get('attempt') or 1) for r in rows), default=0)
        return [r for r in rows if int(r.get('attempt') or 1) == latest]

    def complete(self, trajectory, retry_failed=False):
        rows = self.current_rows(trajectory)
        if any('scenario_complete' in r for r in rows) and not any(str(r.get('scenario_complete')).lower() in {'true','1'} for r in rows):
            return False
        if any(r.get('row_type') == 'failure' for r in rows):
            print(f'{"Retry failed" if retry_failed else "Skip FAILED"} trajectory {trajectory}', flush=True)
            return not retry_failed
        if any('scenario_complete' in r for r in rows):
            return any(str(r.get('scenario_complete')).lower() in {'true','1'} for r in rows)
        return any(r.get('row_type') in {'final','reference'} for r in rows)

    def commit(self, trajectory, rows):
        if not rows:
            raise ValueError('Cannot commit an empty trajectory')
        previous = [r for r in self.rows if r.get('trajectory') == trajectory]
        attempt = max((int(r.get('attempt') or 1) for r in previous), default=0) + 1
        new = [r for r in self.rows if r.get('trajectory') != trajectory]
        for old in previous:
            if old.get('row_type') in {'failure', 'failure_history'}:
                new.append(dict(old, row_type='failure_history', is_final=False,
                                is_current=False, attempt=int(old.get('attempt') or 1)))
        new += [dict(r, trajectory=trajectory, protocol_id=self.fingerprint,
                     attempt=attempt, is_current=True) for r in rows]
        temporary = self.path.with_suffix('.tmp')
        keys = list(dict.fromkeys(k for r in new for k in r))
        try:
            with temporary.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(new)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.path)
            self.rows = new
        finally:
            temporary.unlink(missing_ok=True)

    def __exit__(self, *args):
        os.close(self.fd)
        self.lock.unlink(missing_ok=True)


def terminal_rows(data):
    data = data.copy()
    if "is_current" in data:
        data = data[data.is_current.astype(str).str.lower().isin(["true","1"])]
    if "row_type" in data:
        data = data[data.row_type.isin(["final","failure"])].copy()
    else:
        data = data[data.is_final.astype(str).str.lower().isin(["true","1"])].copy()
    keys = [k for k in ["protocol_id","dataset","distribution","stage","candidate","method","mechanism","target_missing_rate","missing_rate"] if k in data]
    if "rep" not in data: raise ValueError("Independent repetition id required")
    if data.duplicated(keys+["rep"]).any(): raise ValueError("Duplicate terminal records for a repetition")
    return data,keys


def completed_repetitions(rows, methods, mechanism, rate):
    counts = dict.fromkeys(methods, 0)
    if not rows:
        return counts
    final, _ = terminal_rows(pd.DataFrame(rows))
    if final.empty:
        return counts
    final = final[final.status.isin(['completed', 'converged', 'max_iter'])]
    final = final[final.mechanism.eq(mechanism) &
                  pd.to_numeric(final.target_missing_rate, errors='raise').eq(rate)]
    for method in counts:
        reps = pd.to_numeric(final.loc[final.method.eq(method), 'rep'], errors='raise')
        counts[method] = int(reps.nunique())
    return counts


def result_directory(kind):
    from pathlib import Path
    if kind not in {'simulation', 'sensitivity', 'support2', 'testoutput'}:
        raise ValueError(f'Unknown result destination: {kind}')
    folder = 'testoutput' if kind == 'sensitivity' else kind
    path = Path(__file__).resolve().parents[1] / 'results' / folder
    return path


def sensitivity_path(parameter, mechanism='MAR'):
    stems = {'sigma': 'partmeter_sigma', 'l': 'partmeter_L'}
    if parameter not in stems:
        raise ValueError(f'Unknown sensitivity parameter: {parameter}')
    if mechanism not in ('MAR', 'MCAR'):
        raise ValueError(f'Unknown sensitivity mechanism: {mechanism}')
    return result_directory('sensitivity') / f'{stems[parameter]}_{mechanism}_050.csv'
