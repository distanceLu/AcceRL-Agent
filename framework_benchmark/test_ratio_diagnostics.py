import math
import unittest
from types import SimpleNamespace
import torch
from framework_benchmark import ratio_diagnostics as diag


class RatioDiagnosticsTests(unittest.TestCase):
    def tearDown(self):
        diag._rows.clear()

    def test_masks_and_same_token_pairing(self):
        args = SimpleNamespace(context_parallel_size=1, tensor_model_parallel_size=1,
                               pipeline_model_parallel_size=1, advantage_estimator='ppo')
        old = torch.tensor([-4., -100., -2.], requires_grad=True)
        new = torch.tensor([-2., 0., -3.], requires_grad=True)
        adv = torch.tensor([-3., 900., 2.])
        batch = {'loss_masks':[torch.tensor([1,0,1])], 'response_lengths':[3],
                 'rollout_log_probs':[torch.tensor([-4.1,-100.,-2.1])]}
        diag.capture(args,batch,old,new,adv)
        self.assertFalse(diag._rows[0].requires_grad)
        rows = torch.cat([diag._rows[0],torch.zeros((2,2))],1)
        metrics, top = diag.summarize(rows)
        self.assertEqual(metrics['action_tokens'],2)
        self.assertAlmostEqual(metrics['ratio_max'],math.exp(2),places=5)
        self.assertEqual(top[0]['advantage'],-3)
        self.assertEqual(top[0]['old_logprob'],-4)
        self.assertEqual(top[0]['new_logprob'],-2)
        self.assertEqual(top[0]['response_token_offset'],0)
        self.assertIsNone(new.grad)

    def test_global_quantile_uses_tokens_not_rank_means(self):
        rows = torch.zeros((101,10),dtype=torch.float64)
        rows[-1,1] = math.log(1000)
        rows[:,6] = rows[:,1]-rows[:,0]
        rows[:,7] = rows[:,6].exp()
        metrics,_ = diag.summarize(rows)
        self.assertAlmostEqual(metrics['ratio_p99'],1)
        self.assertAlmostEqual(metrics['ratio_max'],1000)
        self.assertAlmostEqual(metrics['new_logprob_mean'],math.log(1000)/101)

    def test_nonfinite_is_visible_and_json_safe(self):
        import json
        rows = torch.zeros((2,10),dtype=torch.float64)
        rows[0,1]=1000
        rows[:,6] = rows[:,1]-rows[:,0]
        rows[:,7] = rows[:,6].exp()
        metrics,top = diag.summarize(rows)
        self.assertEqual(metrics['ratio_nonfinite_count'],1)
        self.assertNotIn('ratio_max',metrics)
        self.assertEqual(top[0]['ratio'],'inf')
        json.dumps({'metrics':metrics,'top':top},allow_nan=False)

if __name__ == '__main__': unittest.main()
