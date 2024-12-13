# plotter.py

import os
import matplotlib.pyplot as plt


class Plotter:
    # Initializes the Plotter with the data required for generating plots
    def __init__(self, metrics_dp, metrics_no_dp, epsilon_values, epochs, save_dir='plots'):
        self.metrics_dp = metrics_dp
        self.metrics_no_dp = metrics_no_dp
        self.epsilon_values = epsilon_values
        self.epochs = epochs
        self.save_dir = save_dir
        self._prepare_save_dir()

    def _prepare_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    # Plot test accuracy
    def plot_test_accuracy(self):
        plt.figure(figsize=(10, 6))
        for epsilon in self.epsilon_values:
            plt.plot(range(1, self.epochs + 1), self.metrics_dp[epsilon]['test_accuracies'], marker='o',
                     label=f'ε = {epsilon}')
        plt.plot(range(1, self.epochs + 1), self.metrics_no_dp['test_accuracies'], marker='x',
                 label='Without DP', color='black')
        plt.title('Test Accuracy vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'test_accuracy_vs_epochs.png'))
        plt.close()

    # Plot training loss
    def plot_training_loss(self):
        plt.figure(figsize=(10, 6))
        for epsilon in self.epsilon_values:
            plt.plot(range(1, self.epochs + 1), self.metrics_dp[epsilon]['train_losses'], marker='o',
                     label=f'ε = {epsilon}')
        plt.plot(range(1, self.epochs + 1), self.metrics_no_dp['train_losses'], marker='x',
                 label='Without DP', color='black')
        plt.title('Training Loss vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'training_loss_vs_epochs.png'))
        plt.close()

    # Plot tradeoff privacy accuracy
    def plot_privacy_accuracy_tradeoff(self):
        final_test_acc_dp = [self.metrics_dp[epsilon]['test_accuracies'][-1] for epsilon in self.epsilon_values]
        final_test_acc_no_dp = self.metrics_no_dp['test_accuracies'][-1]

        plt.figure(figsize=(8, 6))
        plt.plot(self.epsilon_values, final_test_acc_dp, marker='o', label='With DP')
        plt.scatter([self.epsilon_values[-1] + 1], [final_test_acc_no_dp], marker='x',
                    label='Without DP', color='black')
        plt.title('Privacy-Accuracy Trade-off')
        plt.xlabel('ε (Privacy Budget)')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'privacy_accuracy_tradeoff.png'))
        plt.close()

    # Plot privacy budget
    def plot_privacy_budget(self):
        plt.figure(figsize=(10, 6))
        for epsilon in self.epsilon_values:
            plt.plot(range(1, self.epochs + 1), self.metrics_dp[epsilon]['epsilons'], marker='o',
                     label=f'ε = {epsilon}')
        plt.title('Privacy Budget (ε) vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('ε')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, 'privacy_budget_vs_epochs.png'))
        plt.close()

    # Export function
    def generate_all_plots(self):
        self.plot_test_accuracy()
        self.plot_training_loss()
        self.plot_privacy_accuracy_tradeoff()
        self.plot_privacy_budget()
        print(f"\n [DEBUG] All plots have been saved in the folder '{self.save_dir}'")
