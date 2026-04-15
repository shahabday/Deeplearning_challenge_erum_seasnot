from huggingface_hub import PyTorchModelHubMixin
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split, DataLoader
from data_loader import XarraySegmentationDataset


def train_model(
    model, train_loader, test_loader, val_loader, criterion, optimizer, num_epochs
):
    with SummaryWriter() as writer:

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(num_epochs):
            # Training phase
            model.train()  # Set model to training mode
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Calculate training statistics (this part is done for you)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Calculate and store training metrics
            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Validation phase - evaluate on validation set (no weight updates)
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():  # Don't calculate gradients during validation (saves memory)
                for images, labels in val_loader:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Calculate validation metrics
            val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f"Epoch [{epoch+1}/{num_epochs}]:")
            print(
                f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%"
            )
            print(
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
            )
            print("-" * 60)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("acc/train", train_accuracy, epoch)
            writer.add_scalar("acc/val", train_accuracy, epoch)
        writer.add_graph(model, val_loader[0])

    return train_losses, val_losses, train_accuracies, val_accuracies


def main(f_train, f_test, f_val, file="savetheoceans.nc"):
    data = xr.open_dataset(file)

    dataset = XarraySegmentationDataset(data, label_channel="labels", normalize=False)

    if f_train is None:
        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [f_train, f_val, f_test], generator=torch.Generator().manual_seed(42)
    )
    learning_rate = 0.01  # Controls how fast the model learns
    num_epochs = 50  # How many times to go through the training data
    aux_params = dict(
        pooling="max",  # one of 'avg', 'max'
        dropout=0.0,  # dropout ratio, default is None
        activation=torch.nn.ReLU,  # activation function, default is None
        classes=15,  # define number of output labels
    )

    model = smp.Unet(in_channels=11, aux_params=aux_params)
    criterion = DiceLoss(mode="multilabel")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Model created successfully!")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {num_epochs}")
    print("-" * 60)

    # Train the model
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, test_loader, val_loader, criterion, optimizer, num_epochs
    )

    # Plot the results
    # plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

    print("Training complete!")


if __name__ == "__main__":
    main(0.4, 0.5, 0.1)
