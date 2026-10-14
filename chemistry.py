"""RDKit features and graph batching without a graph-learning framework."""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

ELEMENTS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
ATOM_DIM = len(ELEMENTS) + 6
BOND_DIM = 6


def scaffold(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('Invalid SMILES')
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False) or '[ACYCLIC]'


def graph(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not molecule.GetNumAtoms():
        raise ValueError('Invalid or empty molecule')
    atoms = []
    for atom in molecule.GetAtoms():
        number = atom.GetAtomicNum()
        atoms.append([float(number == z) for z in ELEMENTS] + [float(number not in ELEMENTS),
                     atom.GetDegree()/4, atom.GetFormalCharge()/3, float(atom.GetIsAromatic()),
                     atom.GetTotalNumHs()/4, float(atom.IsInRing())])
    edges, bonds = [], []
    kinds = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    for bond in molecule.GetBonds():
        feature = [float(bond.GetBondType() == kind) for kind in kinds] + [
            float(bond.GetIsConjugated()), float(bond.IsInRing())]
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([(a,b), (b,a)]); bonds.extend([feature, feature])
    return (torch.tensor(atoms, dtype=torch.float32),
            torch.tensor(edges, dtype=torch.long).reshape(-1,2).T,
            torch.tensor(bonds, dtype=torch.float32).reshape(-1,BOND_DIM))


def fingerprint(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles)).astype('float32')


def batch_graphs(graphs):
    atoms, edges, bonds, membership, offset = [], [], [], [], 0
    for i, (node, edge, bond) in enumerate(graphs):
        atoms.append(node); edges.append(edge+offset); bonds.append(bond)
        membership.append(torch.full((len(node),), i, dtype=torch.long)); offset += len(node)
    return torch.cat(atoms), torch.cat(edges,1), torch.cat(bonds), torch.cat(membership), len(graphs)
