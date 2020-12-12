import sys
import argparse

from pytorch_lightning.callbacks import LearningRateMonitor

from simple_dataset import *
from precnetlm import *

torch.manual_seed(0)

def run_experiment(dataset_mode, vocab_size, num_batches, epochs, penalize_upper_levels, 
                   num_stacks, a_hat_size, a_hat_layers, r_size, r_layers):
    SEQUENCE_LENGTH = 20
    BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    NUM_BATCHES = num_batches
    
    data_train = SimpleDataset(dataset_mode, vocab_size, SEQUENCE_LENGTH, BATCH_SIZE, NUM_BATCHES)
    data_test = SimpleDataset(dataset_mode, vocab_size, SEQUENCE_LENGTH, BATCH_SIZE, NUM_BATCHES)

    a_hat_stack_sizes=[[a_hat_size for _ in range(a_hat_layers)] for _ in range(num_stacks)]
    r_stack_sizes = [(r_size, r_layers) for _ in range(num_stacks)]

    if penalize_upper_levels:
        mu = torch.FloatTensor([1.0] + [0.1 for _ in range(num_stacks - 1)])
    else:
        mu = torch.FloatTensor([1.0] + [0.1 for _ in range(num_stacks - 1)])

    precnetlm = PreCNetLM(
        vocabs_size=vocab_size,
        a_hat_stack_sizes=a_hat_stack_sizes,
        r_stack_sizes=r_stack_sizes,
        mu=mu
    )

    # print(precnetlm)

    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/')
    lr_logger = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        logger=tb_logger,
        gradient_clip_val=0.25,
        max_epochs=epochs,
        log_every_n_steps=25,
        callbacks=[lr_logger]
    )

    trainer.fit(
        precnetlm, 
        DataLoader(data_train, batch_size=BATCH_SIZE), 
        DataLoader(data_test, batch_size=TEST_BATCH_SIZE),
    )

    # see what the model learned
    bootstrap_run_generator = iter(data_train)

    for run in range(10):
        # prepare the prompt
        x = next(bootstrap_run_generator)
        x = torch.unsqueeze(x, 0)
        bootstrap = []
        for t in range(x.shape[1]):
            c = x[0,t,:].argmax().item()
            bootstrap.append(c)

        # go through the prompt
        _, _, states = precnetlm(x)

        # predict the next characters
        predictions, _, states = precnetlm(
            None, 
            states = states,
            mode='predict',
            predict_next=30
        )

        # decode the characters
        res = []
        for t in range(predictions.shape[1]):
            c = predictions[0,t,:].argmax().item()
            res.append(c)
        
        print(f'Run {run} results:')
        print("Bootstrap:")
        print(bootstrap)
        print("Predictions:")
        print(res)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "-m", "--mode", type=str, help="the kind of the dataset", default="sequence")
    parser.add_argument( "-v", "--vocab_size", type=int, help="size of the vocabulary", default=10)
    parser.add_argument( "-b", "--num_batches", type=int, help="the number of batches to train for", default=10)
    parser.add_argument( "-e", "--epochs", type=int, help="number of epochs to train for", default=10)
    parser.add_argument( '--penalize_upper_levels', dest='penalize_upper_levels', action='store_true')
    parser.set_defaults(penalize_upper_levels=False)
    parser.add_argument( "--num_stacks", type=int, help="number of stacks", default=6)
    parser.add_argument( "--a_hat_size", type=int, help="size of a_hat layer", default=64)
    parser.add_argument( "--a_hat_layers", type=int, help="number of a_hat layers", default=1)
    parser.add_argument( "--r_size", type=int, help="size of r layer", default=64)
    parser.add_argument( "--r_layers", type=int, help="number of r layers", default=1)

    args = parser.parse_args()
    run_experiment(args.mode, args.vocab_size, args.num_batches, args.epochs, 
                   args.penalize_upper_levels, args.num_stacks, args.a_hat_size, 
                   args.a_hat_layers, args.r_size, args.r_layers)
