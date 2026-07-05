import importlib
 import sys
from pathlib import Path

# ensure repo root is on path (so `src` can be imported)
repo_root = next((p for p in [Path.cwd()] + list(Path.cwd().parents) if (p / "src").exists()), Path.cwd())
sys.path.insert(0, str(repo_root))

modules = [
    'src.train',
    'src.api',
    'src.data_preprocessing',
    'src.predict',
    'src.evaluation',
    'src.preprocessing_pipeline',
    'src.utils',
]

ok = []
errs = []
for m in modules:
    try:
        importlib.import_module(m)
        ok.append(m)
    except Exception as e:
        errs.append((m, repr(e)))

print('OK:')
for m in ok:
    print('  ', m)
print('\nErrors:')
for m, e in errs:
    print('  ', m, e)
if errs:
    raise SystemExit(1)
