"""Recompute discrimination and interval-set coverage from held-out predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    predictions=pd.read_csv('results/test-predictions.csv');ledger=pd.read_csv('results/split-ledger.csv')
    metrics=json.loads(Path('results/metrics.json').read_text())
    assert ledger.groupby('scaffold').split.nunique().max()==1
    assert ledger.smiles_sha256.is_unique
    for name,column in [('morgan_logistic','morgan_probability'),('message_passing_ensemble','graph_probability')]:
        values=[]
        for task,rows in predictions.groupby('task'):
            expected=metrics['models'][name]['assays'][task]
            auc=roc_auc_score(rows.label,rows[column]);values.append(auc)
            assert np.isclose(auc,expected['auroc'])
            for method,m in expected['sets'].items():
                q=m['quantile'];prob=rows[column].to_numpy();labels=rows.label.to_numpy()
                sets=np.c_[prob<=q,1-prob<=q];covered=sets[np.arange(len(labels)),labels]
                coverage=pd.DataFrame({'scaffold':rows.scaffold,'covered':covered}).groupby('scaffold').covered.all().mean()
                assert np.isclose(coverage,m['scaffold_simultaneous_coverage'])
                assert np.isclose(covered.mean(),m['row_coverage'])
        assert np.isclose(np.mean(values),metrics['models'][name]['macro_auroc'])
    print('Replayed scaffold disjointness, all 12 assay AUROCs and recorded conformal-set coverage.')


if __name__=='__main__':main()
