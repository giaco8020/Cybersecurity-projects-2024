# main.py

import warnings
from src.data import DataHandler
from src.model import ModelBuilder
from src.trainer import Trainer
from src.plotter import Plotter

import torch


def main():
    # Ignore warnings
    warnings.simplefilter("ignore")

    # Configuration
    DATA_ROOT = '../mnist'  # Directory of MNIST
    MNIST_MEAN = (0.1307,)  # Value for normalizing MNIST images -- default
    MNIST_STD_DEV = (0.3081,)  # Standard deviation for normalizing MNIST images -- default
    STEP_SIZE = 1e-3  # Learning rate -- standard value for this case
    DELTA = 1e-5  # Delta parameter for DP -- standard value for this case
    BATCH_SIZE = 16  # Batch size for data loaders -- we reduced this value due to save computational resources
    CLIPPING_THRESHOLD = 1.2  # Clipping norm for DP training -- standard value for this case

    SAVE_DIR = 'plots'  # Directory to save the generated plots
    EPOCHS = 5  # Number of training EPOCHS
    EPSILON_VALUES = [1, 5, 10]  # List of EPSILON
    NUM_TRAIN_SAMPLES = 10000  # Number of training to use -- we have reduced images to save time and resources
    NUM_TEST_SAMPLES = 2000  # Number of test samples to use -- we have reduced images to save time and resources

    # Device (CPU or GPU if it is present in architecture)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the DataHandler from data.py
    data_handler = DataHandler(
        data_root=DATA_ROOT,
        mnist_mean=MNIST_MEAN,
        mnist_std_dev=MNIST_STD_DEV,
        num_train_samples=NUM_TRAIN_SAMPLES,
        num_test_samples=NUM_TEST_SAMPLES,
        batch_size=BATCH_SIZE
    )
    train_loader, test_loader = data_handler.get_dataloaders()  # Test and Train loaders

    # Dictionary to store metrics for each epsilon value -- usefully for gen plots
    metrics_dp = {}

    # Training with Differential Privacy
    for epsilon in EPSILON_VALUES:
        print(f"\n[DEBUG] Training with ε = {epsilon}")
        # Create a new model for each epsilon to avoid gradient accumulation
        model = ModelBuilder.build_model()
        model = model.to(DEVICE)

        trainer_dp = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=DEVICE,
            dp=True,
            epsilon=epsilon,
            delta=DELTA,
            clipping_threshold=CLIPPING_THRESHOLD,
            step_size=STEP_SIZE,
            epochs=EPOCHS
        )
        trainer_dp.train()
        metrics_dp[epsilon] = trainer_dp.get_metrics()

    # Training without Differential Privacy
    print("\n[DEBUG] Training without Differential Privacy")
    model_no_dp = ModelBuilder.build_model()
    model_no_dp = model_no_dp.to(DEVICE)

    trainer_no_dp = Trainer(
        model=model_no_dp,
        train_loader=train_loader,
        test_loader=test_loader,
        device=DEVICE,
        dp=False,
        step_size=STEP_SIZE,
        epochs=EPOCHS
    )
    trainer_no_dp.train()
    metrics_no_dp = trainer_no_dp.get_metrics()

    # Initialize Plotter and generate plots
    plotter = Plotter(metrics_dp, metrics_no_dp, EPSILON_VALUES, EPOCHS, save_dir=SAVE_DIR)
    plotter.generate_all_plots()


if __name__ == "__main__":
    main()
