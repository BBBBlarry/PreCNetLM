from harry_potter_data_util import *
from lstmlm import *

if __name__ == "__main__":
    SEQUENCE_LENGTH = 100
    BATCH_SIZE = 4
    TEST_BATCH_SIZE = 4
    
    train_path, test_path = prepare_data('data/harry_potter.txt')

    data_train = HarryPotterDataset(train_path, SEQUENCE_LENGTH, BATCH_SIZE)
    data_test = HarryPotterDataset(test_path, SEQUENCE_LENGTH, TEST_BATCH_SIZE)

    vocab_size = data_train.vocab_size()

    hidden_size = 128
    num_layers = 7

    lstmlm = LSTMLM(
        input_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

    print(lstmlm)

    tb_logger = pl_loggers.TensorBoardLogger('lightning_logs/')

    trainer = pl.Trainer(
        logger=tb_logger,
        gradient_clip_val=0.25,
        weights_summary='full',
        max_epochs=5,
        # log_every_n_steps=25,
        # track_grad_norm=2,
        # overfit_batches=0.01,
    )

    trainer.fit(
        lstmlm, 
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
    predictions, state = lstmlm(x)

    # predict the next characters
    PREDICT_NEXT = 30
    for _ in range(PREDICT_NEXT):
        predictions = predictions[:, -1:, :]
        predictions, state = lstmlm(predictions, state)
        char = predictions.argmax(-1)[0]
        res += vocab.ind2voc[char.item()]
    
    print(bootstrap + res)
