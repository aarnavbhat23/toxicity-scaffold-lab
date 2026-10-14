"""Prespecified train/validation/calibration/test scaffold experiment."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from chemistry import batch_graphs, fingerprint, graph, scaffold
from network import ToxicityNetwork, masked_bce

URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz'


def prepare(path):
    frame = pd.read_csv(path); tasks = [c for c in frame if c not in ['mol_id','smiles']]
    excluded, rows, conflicts = [], {}, 0
    for index, row in frame.iterrows():
        molecule = Chem.MolFromSmiles(row.smiles)
        if molecule is None or not 1 <= molecule.GetNumAtoms() <= 128:
            excluded.append({'row': int(index), 'reason': 'unparseable_or_outside_1_128_atoms'}); continue
        canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
        rows.setdefault(canonical, []).append(row[tasks].to_numpy(dtype=float))
    labels, sequences = [], []
    for sequence, values in sorted(rows.items()):
        array = np.array(values); target = []
        for column in array.T:
            unique = np.unique(column[np.isfinite(column)])
            if len(unique) > 1:
                conflicts += 1
            target.append(float(unique[0]) if len(unique) == 1 else float('nan'))
        labels.append(target); sequences.append(sequence)
    groups = np.array([scaffold(s) for s in sequences])
    unique = np.unique(groups); order = np.random.default_rng(912).permutation(unique)
    assignment = {g: ('train' if i < .7*len(order) else 'validation' if i < .8*len(order)
                       else 'calibration' if i < .9*len(order) else 'test') for i,g in enumerate(order)}
    splits = np.array([assignment[g] for g in groups])
    return sequences, np.array(labels,dtype='float32'), tasks, groups, splits, excluded, conflicts


def infer(model, graphs, device):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(*[v.to(device) if isinstance(v,torch.Tensor) else v
                                       for v in batch_graphs(graphs[i:i+64])]).sigmoid().cpu().numpy()
                               for i in range(0,len(graphs),64)])


def quantile(scores, alpha=.1):
    rank = int(np.ceil((len(scores)+1)*(1-alpha)))
    return float(np.sort(scores)[rank-1]) if rank <= len(scores) else 1.


def evaluate(y, prediction, tasks, calibration, test, groups):
    output = {}
    for j, task in enumerate(tasks):
        mask = test & np.isfinite(y[:,j]); cal = calibration & np.isfinite(y[:,j])
        target, prob = y[mask,j].astype(int), prediction[mask,j]
        scores = np.where(y[cal,j] == 1,1-prediction[cal,j],prediction[cal,j])
        row_q = quantile(scores)
        cal_groups = groups[cal]
        group_q = quantile([scores[cal_groups == g].max() for g in np.unique(cal_groups)])
        metrics = {'test_labels': int(mask.sum()), 'positive_fraction': float(target.mean()),
                   'auroc': float(roc_auc_score(target,prob)) if len(np.unique(target)) == 2 else None,
                   'average_precision': float(average_precision_score(target,prob)) if target.any() else None,
                   'brier': float(brier_score_loss(target,prob)), 'sets': {}}
        for name, q in [('row_calibration',row_q),('scaffold_max_calibration',group_q)]:
            sets = np.c_[prob <= q, 1-prob <= q]
            covered = sets[np.arange(len(target)),target]; tg = groups[mask]
            metrics['sets'][name] = {'quantile': q, 'row_coverage': float(covered.mean()),
                'scaffold_simultaneous_coverage': float(np.mean([covered[tg == g].all() for g in np.unique(tg)])),
                'mean_set_size': float(sets.sum(1).mean()),'empty_fraction': float((sets.sum(1)==0).mean())}
        output[task] = metrics
    return output


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data',type=Path,default=Path('data/tox21.csv.gz'))
    parser.add_argument('--device',default='cpu'); parser.add_argument('--epochs',type=int,default=25)
    args = parser.parse_args(); RDLogger.DisableLog('rdApp.warning'); torch.set_num_threads(4)
    started = time.time(); out = Path('results'); out.mkdir(exist_ok=True)
    sequences,y,tasks,groups,splits,excluded,conflicts = prepare(args.data)
    graphs = [graph(s) for s in sequences]; fingerprints = np.array([fingerprint(s) for s in sequences])
    masks = {s: splits == s for s in np.unique(splits)}; train, validation = masks['train'],masks['validation']
    baseline = np.empty_like(y); models = []
    for j in range(len(tasks)):
        mask = train & np.isfinite(y[:,j])
        classifier = LogisticRegression(C=1.,max_iter=1000,random_state=912).fit(fingerprints[mask],y[mask,j])
        baseline[:,j] = classifier.predict_proba(fingerprints)[:,1]
        models.append({'coef': classifier.coef_.tolist(),'intercept': classifier.intercept_.tolist()})
    (out/'fingerprint-model.json').write_text(json.dumps({'tasks':tasks,'models':models}))
    runs, history, selected = [], [], []
    for seed in [912,913]:
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        model = ToxicityNetwork(tasks=len(tasks)).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        best, stale, state, trace = float('inf'),0,None,[]
        for epoch in range(args.epochs):
            model.train(); losses = []; order = rng.permutation(np.flatnonzero(train))
            for offset in range(0,len(order),64):
                indexes = order[offset:offset+64]
                batch = [v.to(args.device) if isinstance(v,torch.Tensor) else v for v in batch_graphs([graphs[i] for i in indexes])]
                optimizer.zero_grad(); loss = masked_bce(model(*batch),torch.tensor(y[indexes],device=args.device))
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
                losses.append(float(loss.detach().cpu()))
            probability = infer(model,[graphs[i] for i in np.flatnonzero(validation)],args.device)
            target = y[validation]; observed = np.isfinite(target); probability = np.clip(probability,1e-7,1-1e-7)
            val = float(-(target[observed]*np.log(probability[observed])+(1-target[observed])*np.log(1-probability[observed])).mean())
            trace.append({'epoch':epoch+1,'train_bce':float(np.mean(losses)),'validation_bce':val})
            print(seed,epoch+1,val,flush=True)
            if val < best:
                best,stale,state = val,0,copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
            else: stale += 1
            if stale >= 5: break
        model.load_state_dict(state); torch.save({'state':state,'tasks':tasks},out/f'graph-{seed}.pt')
        selected.append(int(np.argmin([t['validation_bce'] for t in trace]))+1)
        history.append(trace); runs.append(infer(model,graphs,args.device))
    neural = np.mean(runs,axis=0)
    report = {'source':URL,'sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'source_rows':len(pd.read_csv(args.data)),'unique_molecules':len(sequences),'scaffolds':len(np.unique(groups)),
        'excluded_rows':len(excluded),'conflicting_duplicate_labels_masked':conflicts,
        'split_rows':{s:int(m.sum()) for s,m in masks.items()},'seeds':[912,913], 'selected_epochs':selected,
        'parameters_per_graph_model':sum(p.numel() for p in model.parameters()),'device':args.device,
        'elapsed_seconds':time.time()-started,'models':{}}
    for name,pred in [('morgan_logistic',baseline),('message_passing_ensemble',neural)]:
        metrics = evaluate(y,pred,tasks,masks['calibration'],masks['test'],groups)
        report['models'][name] = {'macro_auroc':float(np.mean([m['auroc'] for m in metrics.values() if m['auroc'] is not None])),
                                'assays':metrics}
    ledger = pd.DataFrame({'molecule_index':np.arange(len(sequences)),'smiles_sha256':[hashlib.sha256(s.encode()).hexdigest() for s in sequences],
                           'scaffold':groups,'split':splits})
    ledger.to_csv(out/'split-ledger.csv',index=False)
    records = []
    for i in np.flatnonzero(masks['test']):
        for j,task in enumerate(tasks):
            if np.isfinite(y[i,j]):
                records.append({'molecule_index':i,'scaffold':groups[i],'task':task,'label':int(y[i,j]),
                                'morgan_probability':float(baseline[i,j]),'graph_probability':float(neural[i,j])})
    pd.DataFrame(records).to_csv(out/'test-predictions.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    (out/'exclusions.json').write_text(json.dumps(excluded,indent=2)+'\n')
    print(json.dumps({k:v['macro_auroc'] for k,v in report['models'].items()}))


if __name__ == '__main__': main()
