from harry_potter_data_util import *
from precnetlm import *

if __name__ == "__main__":
    SEQUENCE_LENGTH = 100
    BATCH_SIZE = 4
    TEST_BATCH_SIZE = 4
    
    train_path, test_path = prepare_data('data/harry_potter.txt')

    data_train = HarryPotterDataset(train_path, SEQUENCE_LENGTH, BATCH_SIZE)
    data_test = HarryPotterDataset(test_path, SEQUENCE_LENGTH, TEST_BATCH_SIZE)

    vocab_size = data_train.vocab_size()

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

    print(precnetlm)

    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/')

    trainer = pl.Trainer(
        logger=tb_logger,
        gradient_clip_val=0.25,
        # weights_summary='full',
        max_epochs=10,
        log_every_n_steps=25,
        # track_grad_norm=2,
        overfit_batches=0.01,
    )

    trainer.fit(
        precnetlm, 
        DataLoader(data_train, batch_size=BATCH_SIZE), 
        DataLoader(data_test, batch_size=TEST_BATCH_SIZE),
    )

    # prepare the prompt
    vocab = data_train.vocab
    bootstrap = 'Knock, knock! '
    res = ""

    x = torch.LongTensor([vocab.voc2ind[c] for c in bootstrap])
    x = torch.stack([F.one_hot(xx, data_train.vocab_size()) for xx in x])
    x = x.type(torch.FloatTensor)
    x = torch.unsqueeze(x, 0)
    
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
    for t in range(predictions.shape[1]):
        c = predictions[0,t,:].argmax().item()
        res += vocab.ind2voc[c]
    
    print(bootstrap + res)
