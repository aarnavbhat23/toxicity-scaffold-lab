import hashlib
from pathlib import Path
import urllib.request
import truststore

URL='https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
SHA256='45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360'


if __name__=='__main__':
    truststore.inject_into_ssl();target=Path('data/tox21.csv.gz');target.parent.mkdir(exist_ok=True)
    content=urllib.request.urlopen(URL,timeout=120).read()
    if hashlib.sha256(content).hexdigest()!=SHA256:raise ValueError('Upstream dataset changed; inspect before updating its pin')
    target.write_bytes(content);print('Verified pinned Tox21 dataset')
