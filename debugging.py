from precnetlm import *

def get_new_simple_model():
    a_hat_stack_sizes=[
        [2], 
        [2], 
        [2], 
    ]
    r_stack_sizes=[
        (2, 1),
        (2, 1),
        (2, 1),
    ]
    vocab_size = 3

    mu = torch.FloatTensor([1.0])

    precnetlm = PreCNetLM(
        vocabs_size=vocab_size,
        a_hat_stack_sizes=a_hat_stack_sizes,
        r_stack_sizes=r_stack_sizes,
        mu=mu
    )

    return precnetlm

def simple_debugging_test():
    # batch of 1 sample, sequence len of 2, and vocab size of 3
    example_input = torch.FloatTensor([
        [
            [1, 0, 0],
            [0, 1, 0],
        ]
    ])
    assert example_input.shape == (1, 2, 3)
    
    model = get_new_simple_model()

    print(model)
    model.forward(example_input)


if __name__ == "__main__":
    simple_debugging_test()