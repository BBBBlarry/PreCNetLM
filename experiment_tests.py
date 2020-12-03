import sys
import getopt

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
        [128, 128], 
        [128, 128], 
        [128, 128], 
    ]
    r_stack_sizes=[
        (128, 1),
        (128, 1),
        (128, 1),
    ]
    mu = torch.FloatTensor([1.0, 0.0, 0.0])

    precnetlm = PreCNetLM(
        vocabs_size=vocab_size,
        a_hat_stack_sizes=a_hat_stack_sizes,
        r_stack_sizes=r_stack_sizes,
        mu=mu
    )

    print(precnetlm)

    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs_debugging/')

    trainer = pl.Trainer(
        logger=tb_logger,
        gradient_clip_val=0.25,
        max_epochs=epochs,
        log_every_n_steps=25,
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
    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv,"hm:v:e:",["mode=", "vocab_size=","epochs="])
    except getopt.GetoptError:
        print('experiment_tests.py -m <mode> -v <vocab_size> -e <epochs>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('experiment_tests.py -m <mode> -v <vocab_size> -e <epochs>')
            sys.exit()
        elif opt in ("-m", "--mode"):
            mode = arg
        elif opt in ("-v", "--vocab_size"):
            vocab_size = int(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)

    run_experiment(mode, vocab_size, epochs)
