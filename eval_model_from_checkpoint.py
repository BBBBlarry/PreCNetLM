import os
import glob

from precnetlm import *
from simple_dataset import *

def load_last_from_checkpoint(log_path, version_number):
    # find all checkpoints
    checkpoint_pattern = os.path.join(log_path, 'default', 
        f'version_{version_number}', 'checkpoints', 'epoch=*.ckpt')
    all_checkpoints = glob.glob(checkpoint_pattern)
    
    if not all_checkpoints:
        raise Exception('No model checkpoints found with {all_checkpoints}')

    # find the latest one
    latest_epoch = 0
    for checkpoint_path in all_checkpoints:
        epoch = int(checkpoint_path.strip(".ckpt").split("epoch=")[-1])
        epoch = max(latest_epoch, epoch)
    
    lastest_checkpoint_path = checkpoint_pattern.replace('*', str(epoch))
    model = PreCNetLM.load_from_checkpoint(lastest_checkpoint_path)
    return model, epoch

if __name__ == "__main__":
    model, epoch = load_last_from_checkpoint("lightning_logs_debugging", 0)

    data_test = SimpleDataset(
        'sequence', 
        vocab_size=40, 
        sequence_length=20, 
        batch_size=4, 
        num_batches=125
    )

    trainer = pl.Trainer()
    test_metrics = trainer.test(
        model, test_dataloaders=DataLoader(data_test, batch_size=4), verbose=False)
    mean_loss = torch.mean(torch.FloatTensor([m['Loss/test'] for m in test_metrics])).item()
    
    entropy = torch.mean(torch.FloatTensor([m['Entropy/test'] for m in test_metrics]))
    perplexity = torch.exp(entropy).item()

    print(f'Test Loss: {mean_loss}')
    print(f'Test perplexity: {perplexity}')
