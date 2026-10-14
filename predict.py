"""Apply released graph and fingerprint models to a SMILES string."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from chemistry import graph,batch_graphs,fingerprint
from network import ToxicityNetwork


def predict(smiles):
    sample=graph(smiles)
    if len(sample[0])>128:raise ValueError('Outside the benchmark atom-count range')
    probabilities=[]
    for seed in [912,913]:
        checkpoint=torch.load(Path('results')/f'graph-{seed}.pt',map_location='cpu',weights_only=True)
        model=ToxicityNetwork(len(checkpoint['tasks'])).eval();model.load_state_dict(checkpoint['state'])
        with torch.no_grad():probabilities.append(model(*batch_graphs([sample])).sigmoid().numpy()[0])
    data=json.loads(Path('results/fingerprint-model.json').read_text());features=fingerprint(smiles)
    baseline=[1/(1+np.exp(-np.dot(np.array(m['coef'])[0],features)-m['intercept'][0])) for m in data['models']]
    return {'graph':dict(zip(data['tasks'],np.mean(probabilities,0).astype(float).tolist())),
            'fingerprint':dict(zip(data['tasks'],map(float,baseline))),
            'note':'Predicted assay activity probabilities, not clinical toxicity, safety or ADMET validation.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('smiles');args=parser.parse_args();print(json.dumps(predict(args.smiles),indent=2))
