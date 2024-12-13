import warnings
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision.datasets import MNIST
from torchvision import models
from opacus.validators import ModuleValidator
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.simplefilter("ignore")


def accuracy(preds, labels):
    return (preds == labels).float().mean().item()


def train_and_evaluate(epsilon, delta=1e-5, epochs=5, batch_size=16, clipping_threshold=1.2):
    STEP_SIZE = 1e-3
    DATA_ROOT = 'mnist'
    MNIST_MEAN = (0.1307,)
    MNIST_STD_DEV = (0.3081,)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD_DEV),
    ])

    train_dataset = MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    NUM_TRAIN_SAMPLES = 10000
    NUM_TEST_SAMPLES = 2000

    train_subset = Subset(train_dataset, list(range(NUM_TRAIN_SAMPLES)))
    test_subset = Subset(test_dataset, list(range(NUM_TEST_SAMPLES)))

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Definizione del modello
    model = models.resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)

    # Validazione e correzione del modello per Opacus
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        model = ModuleValidator.fix(model)
        errors = ModuleValidator.validate(model, strict=False)
        assert not errors, "Il modello non è conforme dopo la correzione."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=STEP_SIZE)

    # Integrazione di Opacus per la DP
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=epsilon,
        target_delta=delta,
        max_grad_norm=clipping_threshold,
    )

    print(f"Usando sigma={optimizer.noise_multiplier} e C={clipping_threshold}")

    # Liste per memorizzare le metriche
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    epsilons = []

    def train(model, train_loader, optimizer, epoch, device):
        model.train()
        epoch_losses = []
        epoch_accuracies = []
        passed = []

        for i, (images, target) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)

            preds = torch.argmax(output, dim=1)
            acc = accuracy(preds, target)

            epoch_losses.append(loss.item())
            epoch_accuracies.append(acc)

            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0 and epoch not in passed:
                current_epsilon = privacy_engine.get_epsilon(delta)
                passed.append(epoch)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(epoch_losses):.6f} "
                    f"Acc@1: {np.mean(epoch_accuracies) * 100:.2f}% "
                    f"(ε = {current_epsilon:.2f}, δ = {delta})"
                )

        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accuracies)
        current_epsilon = privacy_engine.get_epsilon(delta)

        train_losses.append(avg_loss)
        train_accuracies.append(avg_acc)
        epsilons.append(current_epsilon)

        torch.cuda.empty_cache()

    def test(model, test_loader, device):
        model.eval()
        losses = []
        top1_acc = []

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, target in test_loader:
                images = images.to(device)
                target = target.to(device)

                output = model(images)
                loss = criterion(output, target)
                preds = torch.argmax(output, dim=1)
                acc = accuracy(preds, target)

                losses.append(loss.item())
                top1_acc.append(acc)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        top1_avg = np.mean(top1_acc)
        test_accuracies.append(top1_avg)

        print(
            f"\tTest set:"
            f" Loss: {np.mean(losses):.6f} "
            f"Acc: {top1_avg * 100:.2f}%"
        )
        return all_preds, all_labels

    all_preds_list = []
    all_labels_list = []

    for epoch in tqdm(range(epochs), desc=f"Addestramento con ε = {epsilon}", unit="epoch"):
        train(model, train_loader, optimizer, epoch + 1, device)
        all_preds, all_labels = test(model, test_loader, device)
        all_preds_list.append(all_preds)
        all_labels_list.append(all_labels)

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'epsilons': epsilons,
        'all_preds': all_preds,
        'all_labels': all_labels
    }


def train_and_evaluate_no_dp(epochs=5, batch_size=16):
    STEP_SIZE = 1e-3
    DATA_ROOT = '../mnist'
    MNIST_MEAN = (0.1307,)
    MNIST_STD_DEV = (0.3081,)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD_DEV),
    ])

    train_dataset = MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    test_dataset = MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    NUM_TRAIN_SAMPLES = 10000
    NUM_TEST_SAMPLES = 2000

    train_subset = Subset(train_dataset, list(range(NUM_TRAIN_SAMPLES)))
    test_subset = Subset(test_dataset, list(range(NUM_TEST_SAMPLES)))

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Definizione del modello
    model = models.resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)

    # Validazione e correzione del modello per Opacus
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        model = ModuleValidator.fix(model)
        errors = ModuleValidator.validate(model, strict=False)
        assert not errors, "Il modello non è conforme dopo la correzione."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=STEP_SIZE)

    print("Addestramento senza Differential Privacy")

    # Liste per memorizzare le metriche
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    def train(model, train_loader, optimizer, epoch, device):
        model.train()
        epoch_losses = []
        epoch_accuracies = []

        for i, (images, target) in enumerate(train_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)

            preds = torch.argmax(output, dim=1)
            acc = accuracy(preds, target)

            epoch_losses.append(loss.item())
            epoch_accuracies.append(acc)

            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(epoch_losses):.6f} "
                    f"Acc@1: {np.mean(epoch_accuracies) * 100:.2f}%"
                )

        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accuracies)

        train_losses.append(avg_loss)
        train_accuracies.append(avg_acc)

        torch.cuda.empty_cache()

    def test(model, test_loader, device):
        model.eval()
        losses = []
        top1_acc = []

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, target in test_loader:
                images = images.to(device)
                target = target.to(device)

                output = model(images)
                loss = criterion(output, target)
                preds = torch.argmax(output, dim=1)
                acc = accuracy(preds, target)

                losses.append(loss.item())
                top1_acc.append(acc)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        top1_avg = np.mean(top1_acc)
        test_accuracies.append(top1_avg)

        print(
            f"\tTest set:"
            f" Loss: {np.mean(losses):.6f} "
            f"Acc: {top1_avg * 100:.2f}%"
        )
        return all_preds, all_labels

    all_preds_list = []
    all_labels_list = []

    for epoch in tqdm(range(epochs), desc="Addestramento senza DP", unit="epoch"):
        train(model, train_loader, optimizer, epoch + 1, device)
        all_preds, all_labels = test(model, test_loader, device)
        all_preds_list.append(all_preds)
        all_labels_list.append(all_labels)

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'all_preds': all_preds,
        'all_labels': all_labels
    }


# Definire diversi valori di epsilon
epsilon_values = [1, 5, 10]
delta = 1e-5
epochs = 5
batch_size = 16
clipping_threshold = 1.2

# Dizionario per memorizzare le metriche per ciascun epsilon
metrics_dp = {}

for epsilon in epsilon_values:
    print(f"\nAddestramento con ε = {epsilon}")
    metrics_dp[epsilon] = train_and_evaluate(
        epsilon=epsilon,
        delta=delta,
        epochs=epochs,
        batch_size=batch_size,
        clipping_threshold=clipping_threshold
    )

# Addestrare il modello senza DP
metrics_no_dp = train_and_evaluate_no_dp(
    epochs=epochs,
    batch_size=batch_size
)


# Funzione per tracciare e salvare i grafici
def plot_and_save(metrics_dp, metrics_no_dp, epsilon_values, epochs, save_dir='plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. Accuratezza di Test vs Epoche
    plt.figure(figsize=(10, 6))
    for epsilon in epsilon_values:
        plt.plot(range(1, epochs + 1), metrics_dp[epsilon]['test_accuracies'], marker='o', label=f'ε = {epsilon}')
    plt.plot(range(1, epochs + 1), metrics_no_dp['test_accuracies'], marker='x', label='Senza DP', color='black')
    plt.title('Accuratezza di Test vs Epoche')
    plt.xlabel('Epoca')
    plt.ylabel('Acc@1')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'test_accuracy_vs_epochs.png'))
    plt.close()

    # 2. Perdita di Addestramento vs Epoche
    plt.figure(figsize=(10, 6))
    for epsilon in epsilon_values:
        plt.plot(range(1, epochs + 1), metrics_dp[epsilon]['train_losses'], marker='o', label=f'ε = {epsilon}')
    plt.plot(range(1, epochs + 1), metrics_no_dp['train_losses'], marker='x', label='Senza DP', color='black')
    plt.title('Perdita di Addestramento vs Epoche')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_loss_vs_epochs.png'))
    plt.close()

    # 3. Trade-off Privacy-Accuratezza
    final_test_acc_dp = [metrics_dp[epsilon]['test_accuracies'][-1] for epsilon in epsilon_values]
    final_test_acc_no_dp = metrics_no_dp['test_accuracies'][-1]

    plt.figure(figsize=(8, 6))
    plt.plot(epsilon_values, final_test_acc_dp, marker='o', label='Con DP')
    plt.scatter([epsilon_values[-1] + 1], [final_test_acc_no_dp], marker='x', label='Senza DP',
                color='black')  # Posizionare senza DP come punto
    plt.title('Trade-off Privacy-Accuratezza')
    plt.xlabel('ε (Budget di Privacy)')
    plt.ylabel('Acc@1')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'privacy_accuracy_tradeoff.png'))
    plt.close()

    # 4. Budget di Privacy (ε) vs Epoche
    plt.figure(figsize=(10, 6))
    for epsilon in epsilon_values:
        plt.plot(range(1, epochs + 1), metrics_dp[epsilon]['epsilons'], marker='o', label=f'ε = {epsilon}')
    plt.title('Budget di Privacy (ε) vs Epoche')
    plt.xlabel('Epoca')
    plt.ylabel('ε')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'privacy_budget_vs_epochs.png'))
    plt.close()

    print(f"\nTutti i grafici richiesti sono stati salvati nella cartella '{save_dir}'.")


# Eseguire il training e generare i grafici
plot_and_save(metrics_dp, metrics_no_dp, epsilon_values, epochs, save_dir='src/plots')