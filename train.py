import torch
from torch.utils.data import DataLoader
from autoencoder import *

class AutoencoderTrainer:
    def __init__(self, model: VanillaAutoEncoder, trainpath: str, testpath: str, batch_size: int):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.train_data = torch.load(trainpath)
        self.test_data = torch.load(testpath)
        # TODO need to define dataset to use or will tensors just go brr?
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size)
        self.batch_size = batch_size
        self.test_loader = DataLoader(self.test_data, batch_size=self.test_data.shape[0])

    def train_one_epoch(self, epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Compute the loss and its gradients
            loss = self.model.loss(inputs)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % self.batch_size == self.batch_size - 1:
                last_loss = running_loss / self.batch_size  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.train_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss