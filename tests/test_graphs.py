import unittest
import torch
from chemistry import graph, batch_graphs, scaffold
from network import ToxicityNetwork, masked_bce
from experiment import quantile


class GraphInvariants(unittest.TestCase):
    def test_no_cross_molecule_edges(self):
        batch = batch_graphs([graph('CCO'),graph('c1ccccc1')])
        _,edges,_,membership,_ = batch
        self.assertTrue(torch.equal(membership[edges[0]],membership[edges[1]]))

    def test_batch_is_invisible(self):
        torch.manual_seed(4); model = ToxicityNetwork().eval()
        with torch.no_grad():
            single = model(*batch_graphs([graph('CCO')]))[0]
            together = model(*batch_graphs([graph('CCO'),graph('CC')]))[0]
        torch.testing.assert_close(single,together)

    def test_missing_targets_have_no_gradient(self):
        logits = torch.tensor([[1.,2.]],requires_grad=True)
        masked_bce(logits,torch.tensor([[1.,float('nan')]])).backward()
        self.assertEqual(logits.grad[0,1].item(),0.)

    def test_atom_order_is_invisible(self):
        torch.manual_seed(8);model=ToxicityNetwork().eval()
        with torch.no_grad():
            forward=model(*batch_graphs([graph('CCO')]))
            reversed_order=model(*batch_graphs([graph('OCC')]))
        torch.testing.assert_close(forward,reversed_order)

    def test_related_ring_scaffolds(self):
        self.assertEqual(scaffold('Cc1ccccc1'),scaffold('Oc1ccccc1'))
        self.assertEqual(scaffold('CCO'),'[ACYCLIC]')

    def test_small_calibration_abstains(self):
        self.assertEqual(quantile([.2]),1.)


if __name__ == '__main__': unittest.main()
