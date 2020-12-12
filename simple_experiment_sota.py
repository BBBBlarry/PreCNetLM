import sys
import argparse

from pytorch_lightning.callbacks import LearningRateMonitor

from simple_dataset import *
from lstmlm import *

torch.manual_seed(0)

def run_experiment(dataset_mode, vocab_size, num_batches, epochs, 
                   hidden_size, num_layers):
    SEQUENCE_LENGTH = 20
    BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    NUM_BATCHES = num_batches
    
    data_train = SimpleDataset(dataset_mode, vocab_size, SEQUENCE_LENGTH, BATCH_SIZE, NUM_BATCHES)
    data_test = SimpleDataset(dataset_mode, vocab_size, SEQUENCE_LENGTH, BATCH_SIZE, NUM_BATCHES)

    lstmlm = LSTMLM(
        input_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

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
        lstmlm, 
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
        predictions, state = lstmlm(x)

        # predict the next characters
        PREDICT_NEXT = 30
        res = []
        for _ in range(PREDICT_NEXT):
            predictions = predictions[:, -1:, :]
            predictions, state = lstmlm(predictions, state)
            char = predictions.argmax(-1)[0]
            res.append(char.item())
        
       
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
    parser.add_argument( "--hidden_size", type=int, help="hidden size for LSTM", default=64)
    parser.add_argument( "--num_layers", type=int, help="number of layers for LSTM", default=3)
    
    args = parser.parse_args()
    run_experiment(args.mode, args.vocab_size, args.num_batches, args.epochs,
                   args.hidden_size, args.num_layers)
