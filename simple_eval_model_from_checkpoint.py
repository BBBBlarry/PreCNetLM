import argparse
import glob
import os

from precnetlm import *
from simple_dataset import *


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
    model = PreCNetLM.load_from_checkpoint(lastest_checkpoint_path)
    return model, epoch

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
        "-s", 
        "--version", 
        type=int,
        help="the version of the model to run"
    )
    parser.set_defaults(penalize_upper_levels=False)
    args = parser.parse_args()
    
    SEQUENCE_LENGTH = 20
    BATCH_SIZE = 16
    VOCAB_SIZE = args.vocab_size
    NUM_BATCHES = VOCAB_SIZE
    MODE = args.mode
    VERSION = args.version
    
    assert VERSION is not None

    model, epoch = load_last_from_checkpoint("lightning_logs_debugging", VERSION)

    data_test = SimpleDataset(
        mode=MODE,
        vocab_size=VOCAB_SIZE, 
        sequence_length=SEQUENCE_LENGTH, 
        batch_size=BATCH_SIZE, 
        num_batches=100
    )

    trainer = pl.Trainer()
    test_metrics = trainer.test(
        model, test_dataloaders=DataLoader(data_test, batch_size=4), verbose=False)
    mean_loss = torch.mean(torch.FloatTensor([m['Loss/test'] for m in test_metrics])).item()
    
    entropy = torch.mean(torch.FloatTensor([m['Entropy/test'] for m in test_metrics]))
    perplexity = torch.exp(entropy).item()

    print(f'Test Loss: {mean_loss}')
    print(f'Test perplexity: {perplexity}')
