"""
training_pipeline.py : pipeline to train and test the model

This module has functions that trains and tests the models 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np

from finalproject.nn_models import MobileNetV3SmallCNN

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training_epoch(training_params, model, device, train_loader, optimizer, epoch):
    """
    Executes one training epoch, processing batches of data, computing loss,
    and updating model weights.

    This function iterates through the `train_loader`, computes the loss for each batch,
    performs backpropagation, and updates the model weights using the provided optimizer.
    It tracks the accuracy and loss over the course of the epoch and prints periodic logs.

    Args:
        training_params (Namespace or dict): Training configuration parameters
        (e.g., `log_interval`, `dry_run`).
        model (torch.nn.Module): The model to train.
        device (torch.device): The device (CPU or GPU) on which to perform computations.
        train_loader (torch.utils.data.DataLoader): DataLoader providing batches of training data.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model weights.
        epoch (int): The current epoch number.

    Returns:
        tuple: A tuple containing:
            - `loss` (float): The final loss value for the last batch of the epoch.
            - `train_accuracy` (float): The accuracy of the model on the training set
            for this epoch.

    Raises:
        ValueError: If the model output is not compatible with the expected loss function.
    """
    model.train()
    correct = 0
    total = 0

    for batch_idx, data in enumerate(train_loader):
        inputs, targets = data["image_input"].float(), data["target"].long()
        data, target = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Use CrossEntropyLoss if MobileNetV3Small, else use F.nll_loss
        if isinstance(model, MobileNetV3SmallCNN):
            loss = nn.CrossEntropyLoss()(output, target)
        else:
            # Assuming model outputs log probabilities for NLL loss
            loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % training_params.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if training_params.dry_run:
                break
    train_accuracy = correct / total
    return loss.item(), train_accuracy


def predict(model, device, test_loader):
    """
    Evaluates the model on the test dataset and computes the average loss and accuracy.

    This function sets the model to evaluation mode, iterates through the `test_loader` to
    compute the loss and accuracy for the test set, and returns the average test loss
    and accuracy. It uses CrossEntropyLoss for MobileNetV3Small model, and NLLLoss for
    other models assuming they output log probabilities.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        test_loader (torch.utils.data.DataLoader): DataLoader providing batches of test data.

    Returns:
        tuple: A tuple containing:
            - `test_loss` (float): The average loss over the entire test set.
            - `test_accuracy` (float): The accuracy of the model on the test set.

    Raises:
        ValueError: If the model's output is incompatible with the expected loss function
        (i.e., neither `CrossEntropyLoss` nor `NLLLoss`).

    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data["image_input"], data["target"].long()
            data, target = inputs.to(device), targets.to(device)
            output = model(data)

            # Use CrossEntropyLoss if MobileNetV3Small, else use F.nll_loss
            if isinstance(model, MobileNetV3SmallCNN):
                test_loss += nn.CrossEntropyLoss(reduction="sum")(output, target).item()
            else:
                # Assuming model outputs log probabilities for NLL loss
                test_loss += F.nll_loss(output, target, reduction="sum").item()

            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / total

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_loss, test_accuracy


def configure_training_device(training_params):
    """
    Configures the device (CPU, CUDA, or MPS) for training and sets up batch size parameters.

    This function checks the available hardware (GPU with CUDA, MPS for Apple devices, or CPU)
    and configures the device accordingly. It also sets the manual seed for reproducibility
    and prepares the training and testing DataLoader parameters based on the chosen device.

    Args:
        training_params (Namespace or dict): Configuration parameters for training.
            Expected fields include:
            - `no_cuda` (bool): If True, CUDA will not be used, even if available.
            - `no_mps` (bool): If True, MPS (Metal Performance Shaders) will not be used,
            even if available.
            - `seed` (int): Random seed for reproducibility.
            - `batch_size` (int): The batch size for training.
            - `test_batch_size` (int): The batch size for testing.

    Returns:
        tuple: A tuple containing:
            - `device` (torch.device): The device to be used for training and inference
            (CPU, CUDA, or MPS).
            - `train_kwargs` (dict): A dictionary of training DataLoader arguments,
            including batch size,
              and additional CUDA-specific arguments if applicable.
            - `test_kwargs` (dict): A dictionary of testing DataLoader arguments,
            including batch size, and additional CUDA-specific arguments if applicable.

    """
    use_cuda = not training_params.no_cuda and torch.cuda.is_available()
    use_mps = not training_params.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(training_params.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": training_params.batch_size}
    test_kwargs = {"batch_size": training_params.test_batch_size}

    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    return device, train_kwargs, test_kwargs


def train_modified_scheduler_with_early_stopping(
    model, training_params, device, train_loader, test_loader, model_name, patience=4
):
    """
    Trains the model using a specified optimizer and learning rate scheduler, with early stopping.

    This function trains the model for a specified number of epochs, applying a learning rate
    scheduler and using early stopping if the test accuracy does not improve for a given
    number of epochs (patience). It tracks training and testing loss and accuracy, and saves
    the model whenever there is an improvement in performance.

    Args:
        model (torch.nn.Module): The model to train.
        training_params (Namespace or dict): Training configuration parameters. Expected fields:
            - `optimizer_type` (str): The optimizer type to use ('adamw' or 'adadelta').
            - `lr` (float): Learning rate for the optimizer.
            - `weight_decay` (float): Weight decay for the optimizer (for 'adadelta').
            - `step_size` (int): Step size for the learning rate scheduler.
            - `gamma` (float): Gamma value for the learning rate scheduler.
            - `epochs` (int): Number of epochs to train.
            - `save_model` (bool): If True, the model will be saved after each epoch with
            improved performance.
        device (torch.device): The device (CPU or GPU) to use for training.
        train_loader (torch.utils.data.DataLoader): DataLoader providing batches of training data.
        test_loader (torch.utils.data.DataLoader): DataLoader providing batches of test data.
        model_name (str): The name used to save the model's state_dict if performance improves.
        patience (int, optional): The number of epochs to wait for an improvement in test
        accuracy before triggering early stopping. Default is 4.

    Returns:
        tuple: A tuple containing:
            - `train_loss` (np.ndarray): Array of training loss values for each epoch.
            - `test_loss` (np.ndarray): Array of test loss values for each epoch.
            - `train_accuracies` (np.ndarray): Array of training accuracies for each epoch.
            - `test_accuracies` (np.ndarray): Array of test accuracies for each epoch.
            - `model` (torch.nn.Module): The trained model after early stopping or
            completion of all epochs.

    Raises:
        ValueError: If an unsupported optimizer type is specified in
        `training_params.optimizer_type`.
    """
    # Select optimizer based on the training_params.optimizer_type
    if training_params.optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=training_params.lr)
    elif training_params.optimizer_type.lower() == "adadelta":
        optimizer = optim.Adadelta(
            model.parameters(),
            lr=training_params.lr,
            weight_decay=training_params.weight_decay,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer type: {training_params.optimizer_type}"
        )

    scheduler = StepLR(
        optimizer, step_size=training_params.step_size, gamma=training_params.gamma
    )

    best_test_accuracy = 0.0
    epochs_without_improvement = 0

    train_loss = np.array([], dtype=np.float32)
    test_loss = np.array([], dtype=np.float32)
    train_accuracies = np.array([], dtype=np.float32)
    test_accuracies = np.array([], dtype=np.float32)

    for epoch in range(1, training_params.epochs + 1):
        train_loss_epoch, train_accuracy = run_training_epoch(
            training_params, model, device, train_loader, optimizer, epoch
        )
        test_loss_epoch, test_accuracy = predict(model, device, test_loader)

        scheduler.step()

        train_loss = np.append(train_loss, train_loss_epoch)
        test_loss = np.append(test_loss, test_loss_epoch)
        train_accuracies = np.append(train_accuracies, train_accuracy)
        test_accuracies = np.append(test_accuracies, test_accuracy)

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            epochs_without_improvement = 0  # reset counter if we get better accuracy
        else:
            epochs_without_improvement += 1

        # If there is no improvement for 'patience' epochs, stop training
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

        if training_params.save_model:
            torch.save(model.state_dict(), model_name + ".pt")

    return train_loss, test_loss, train_accuracies, test_accuracies, model
