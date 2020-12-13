import argparse
import glob
import os

from lstmlm import *
from twitter_data_util import *


def load_last_from_checkpoint(log_path, version_number):
    # find all checkpoints
    checkpoint_pattern = os.path.join(log_path, 'default', 
        f'version_{version_number}', 'checkpoints', 'epoch=*.ckpt')
    all_checkpoints = glob.glob(checkpoint_pattern)
    
    if not all_checkpoints:
        raise Exception(f'No model checkpoints found with {checkpoint_pattern}')

    # find the latest one
    latest_epoch = 0
    for checkpoint_path in all_checkpoints:
        epoch = int(checkpoint_path.strip(".ckpt").split("epoch=")[-1])
        epoch = max(latest_epoch, epoch)
    
    lastest_checkpoint_path = checkpoint_pattern.replace('*', str(epoch))
    model = LSTMLM.load_from_checkpoint(lastest_checkpoint_path)
    return model, epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", 
        "--version", 
        type=int,
        help="the version of the model to run"
    )
    parser.set_defaults(penalize_upper_levels=False)
    args = parser.parse_args()

    SEQUENCE_LENGTH = 100
    BATCH_SIZE = 4
    test_path = DATA_PATH + 'twitter_chars_test.pkl'
    data_test = TwitterDataset(test_path, SEQUENCE_LENGTH, BATCH_SIZE)
    VOCAB_SIZE = data_test.vocab_size()
    VERSION = args.version
    
    assert VERSION is not None

    model, epoch = load_last_from_checkpoint("lightning_logs", VERSION)

    trainer = pl.Trainer()
    test_metrics = trainer.test(
        model, test_dataloaders=DataLoader(data_test, batch_size=BATCH_SIZE), verbose=False)
    mean_loss = torch.mean(torch.FloatTensor([m['Loss/test'] for m in test_metrics]))
    perplexity = torch.exp(mean_loss)

    print(f'Test Loss: {mean_loss.item()}')
    print(f'Test perplexity: {perplexity.item()}')
