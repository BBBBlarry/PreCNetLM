import unittest 

from precnetlm import *

class TestPreCNetLMModel(unittest.TestCase):
    def test_no_mixing_data_gradient_on_output(self):
        # examine the gradient of the n-th minibatch sample w.r.t. all inputs
        n = 3  
        num_samples = 10

        # 1. require gradient on input batch
        example_input = torch.rand(num_samples, 20, 64, requires_grad=True)

        # 2. run batch through model
        model = self.get_new_model(64)
        output, _, _ = model(example_input)

        # 3. compute a dummy loss on n-th output sample and back-propagate
        output[n].abs().sum().backward()

        # 4. check that gradient on samples i != n are zero!
        # sanity check: if this does not return 0, you have a bug!
        for i in range(num_samples):
            if i != n:
                self.assertEqual(example_input.grad[i].abs().sum().item(), 0.0)

    def test_no_mixing_data_gradient_on_error(self):
        # examine the gradient of the n-th minibatch sample w.r.t. all inputs
        n = 3
        num_samples = 10

        # 1. require gradient on input batch
        example_input = torch.rand(num_samples, 20, 64, requires_grad=True)

        # 2. run batch through model
        model = self.get_new_model(64)
        _, errors, _ = model(example_input)


        # 3. compute a dummy loss on n-th output sample and back-propagate
        loss = 0.0
        for level in errors:
            sum_e = errors[level][n, 1:, :].mean()
            loss += 1.0 * sum_e
        loss.backward()

        # 4. check that gradient on samples i != n are zero!
        # sanity check: if this does not return 0, you have a bug!
        for i in range(num_samples):
            if i != n:
                self.assertEqual(example_input.grad[i].abs().sum().item(), 0.0)

    def get_new_model(self, vocab_size = 64):
        a_hat_stack_sizes=[
            [128, 128], 
            [128, 128], 
            [128, 128], 
        ]
        r_stack_sizes=[
            (128, 1),
            (128, 1),
            (128, 1),
        ]

        mu = torch.FloatTensor([1.0, 0.01, 0.01])

        precnetlm = PreCNetLM(
            vocabs_size=vocab_size,
            a_hat_stack_sizes=a_hat_stack_sizes,
            r_stack_sizes=r_stack_sizes,
            mu=mu
        )

        return precnetlm

if __name__ == "__main__":
    unittest.main()
