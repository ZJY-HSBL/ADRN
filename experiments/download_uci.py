"""Download official UCI bank and default datasets. GMSC requires Kaggle data."""
import io,zipfile,urllib.request
from pathlib import Path
root=Path('data/raw');root.mkdir(parents=True,exist_ok=True)
def get(url):
    with urllib.request.urlopen(url,timeout=120) as r:return r.read()
outer=zipfile.ZipFile(io.BytesIO(get('https://archive.ics.uci.edu/static/public/222/bank+marketing.zip')))
inner=zipfile.ZipFile(io.BytesIO(outer.read('bank.zip')))
(root/'bank-full.csv').write_bytes(inner.read('bank-full.csv'))
z=zipfile.ZipFile(io.BytesIO(get('https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip')))
(root/'default.xls').write_bytes(z.read('default of credit card clients.xls'))
print('Downloaded UCI datasets; place Kaggle cs-training.csv in data/raw/.')
