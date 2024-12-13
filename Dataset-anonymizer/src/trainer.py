# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import numpy as np
from tqdm import tqdm


# Computes the accuracy of predictions
def accuracy(preds, labels):
    return (preds == labels).float().mean().item()


class Trainer:
    def __init__(self, model, train_loader, test_loader, device,
                 dp=False, epsilon=None, delta=1e-5, clipping_threshold=1.2,
                 step_size=1e-3, epochs=5):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.dp = dp
        self.epsilon = epsilon
        self.delta = delta
        self.clipping_threshold = clipping_threshold
        self.step_size = step_size
        self.epochs = epochs
        self.criterion = nn.CrossEntropyLoss() # Loss function
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.step_size)
        self.privacy_engine = None
        if self.dp and self.epsilon is not None:
            self._enable_dp()

        # Lists to track training metrics
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.epsilons = []

    # Enables DP for the specific model
    def _enable_dp(self):
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            epochs=self.epochs,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            max_grad_norm=self.clipping_threshold,
        )
        print(f"[DEBUG] Using sigma={self.optimizer.noise_multiplier} and C={self.clipping_threshold}")

    # Trains the model for one epoch.
    def _train_epoch(self, epoch):
        self.model.train()
        epoch_losses = []
        epoch_accuracies = []
        passed = []

        for i, (images, targets) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            images, targets = images.to(self.device), targets.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            acc = accuracy(preds, targets)

            epoch_losses.append(loss.item())
            epoch_accuracies.append(acc)

            if (i + 1) % 50 == 0 and (epoch not in passed):
                if self.dp:
                    current_epsilon = self.privacy_engine.get_epsilon(self.delta)
                    print(
                        f"[DEBUG] "
                        f"\tTrain Epoch: {epoch} [{i + 1}/{len(self.train_loader)}] \t"
                        f"Loss: {np.mean(epoch_losses):.6f} "
                        f"Accuracy: {np.mean(epoch_accuracies) * 100:.2f}% "
                        f"(ε = {current_epsilon:.2f}, δ = {self.delta})"
                    )
                    passed.append(epoch)
                else:
                    print(
                        f"[DEBUG] "
                        f"\tTrain Epoch: {epoch} [{i + 1}/{len(self.train_loader)}] \t"
                        f"Loss: {np.mean(epoch_losses):.6f} "
                        f"Accuracy: {np.mean(epoch_accuracies) * 100:.2f}%"
                    )

        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accuracies)
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_acc)
        if self.dp:
            current_epsilon = self.privacy_engine.get_epsilon(self.delta)
            self.epsilons.append(current_epsilon)
        torch.cuda.empty_cache()

    # Evaluates the model on the test dataset.
    def _test_epoch(self):
        self.model.eval()
        losses = []
        accuracies = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, targets in self.test_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                preds = torch.argmax(outputs, dim=1)
                acc = accuracy(preds, targets)

                losses.append(loss.item())
                accuracies.append(acc)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        avg_loss = np.mean(losses)
        avg_acc = np.mean(accuracies)
        self.test_accuracies.append(avg_acc)
        print(
            f"[DEBUG] "
            f"\tTest set: Loss: {avg_loss:.6f} Accuracy: {avg_acc * 100:.2f}%"
        )
        return all_preds, all_labels

    # Executes the full training process for the specified number of epochs.
    def train(self):
        for epoch in tqdm(range(1, self.epochs + 1), desc="Training", unit="epoch"):
            self._train_epoch(epoch)
            self._test_epoch()

    def get_metrics(self):
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'epsilons': self.epsilons if self.dp else None
        }
