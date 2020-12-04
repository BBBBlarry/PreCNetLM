import sys
import argparse

from pytorch_lightning.callbacks import LearningRateMonitor

from test_data_util import *
from precnetlm import *


def run_experiment(dataset_mode, vocab_size, epochs):
    SEQUENCE_LENGTH = 20
    BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    NUM_BATCHES = vocab_size
    
    data_train = TestDataset(dataset_mode, vocab_size, SEQUENCE_LENGTH, BATCH_SIZE, NUM_BATCHES)
    data_test = TestDataset(dataset_mode, vocab_size, SEQUENCE_LENGTH, BATCH_SIZE, NUM_BATCHES)

    a_hat_stack_sizes=[
        [64], 
        [64], 
        [64], 
    ]
    r_stack_sizes=[
        (64, 1),
        (64, 1),
        (64, 1),
    ]
    mu = torch.FloatTensor([1.0, 0.1, 0.1])

    precnetlm = PreCNetLM(
        vocabs_size=vocab_size,
        a_hat_stack_sizes=a_hat_stack_sizes,
        r_stack_sizes=r_stack_sizes,
        mu=mu
    )

    print(precnetlm)

    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs_debugging/')
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
    parser.add_argument(
        "-m", 
        "--mode", 
        type=str,
        help="the kind of the dataset",
        default="sequence"
    )
    parser.add_argument(
        "-v", 
        "--vocab_size", 
        type=int,
        help="size of the vocabulary",
        default=10
    )
    parser.add_argument(
        "-e", 
        "--epochs", 
        type=int,
        help="number of epochs to train for",
        default=10
    )
    
    args = parser.parse_args()
    run_experiment(args.mode, args.vocab_size, args.epochs)
