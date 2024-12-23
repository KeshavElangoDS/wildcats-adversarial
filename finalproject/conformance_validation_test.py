"""
conformance_validation_test.py: Implements conformance and input, output validation on model.

This module contains functions that implement the conformance testing and the input , output
validation on model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from finalproject.training_pipeline import predict, configure_training_device
from finalproject.training_pipeline import train_modified_scheduler_with_early_stopping

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model_conformance(model, train_loader, test_loader, device):
    """
    Tests the conformance of a model by validating its input/output shapes, loss, parameters,
    and test accuracy.

    This function performs a series of checks to ensure that the model behaves as expected during
    training and testing. It verifies the following:
    - The input image has 3 channels (RGB).
    - The model produces an output with 10 classes.
    - The loss is non-negative.
    - The model has trainable parameters.
    - The test accuracy is greater than 0%.

    Args:
        model (torch.nn.Module): The model to be tested.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (torch.device): The device (CPU or GPU) on which the model should be tested.

    Raises:
        AssertionError: If any of the following conditions are not met:
            - The input has 3 channels.
            - The output has 10 classes.
            - The loss is non-negative.
            - The model has parameters to train.
            - The test accuracy is greater than 0%.

    Example:
        >>> model = YourModel()
        >>> train_loader = your_train_loader
        >>> test_loader = your_test_loader
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> test_model_conformance(model, train_loader, test_loader, device)
    """

    # Get a sample batch from the train loader
    for data in train_loader:
        inputs, targets = data["image_input"].float(), data["target"].long()
        inputs, targets = inputs.to(device), targets.to(device)
        break  # Use only the first batch

    # Check the input shape: Ensure it has 3 channels (RGB)
    assert (
        inputs.shape[1] == 3
    ), "Expected 3 channels in input image, but got {inputs.shape[1]} channels."

    output = model(inputs)

    assert (
        output.shape[1] == 10
    ), f"Expected 10 output classes, but got {output.shape[1]} classes."

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, targets)

    assert loss.item() >= 0, f"Loss should be non-negative, but got {loss.item()}."

    # Check that the model has trainable parameters
    initial_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the model: {initial_params}")

    assert initial_params > 0, "Model has no parameters to train."

    _, test_accuracy = predict(model, device, test_loader)
    assert test_accuracy > 0.0, "Test accuracy should be greater than 0%."

    print("All conformance tests passed successfully!")


def test_model_conformance_only(
    model, training_params, device, train_loader, test_loader
):
    """
    Validates the model's conformance by testing its input/output shapes, loss, and model parameters.

    This function performs conformance checks to ensure that the model behaves as expected, validating:
    - The input image has 3 channels (RGB).
    - The model produces an output with the expected number of classes.
    - The loss is non-negative.
    - The model has trainable parameters.
    - The test accuracy is greater than 0%.

    Conformance testing is done without training or testing the model. It is useful to ensure the model
    is properly set up and behaves as expected.

    Args:
        model (torch.nn.Module): The model to be validated.
        training_params (TrainingParameters): The parameters for training, though they are not used in this version.
        device (torch.device): The device (CPU or GPU) to run the model on.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset, used for conformance checks.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset, used for conformance checks.

    Raises:
        AssertionError: If any of the following conditions are not met during the conformance checks:
            - The input has 3 channels.
            - The output has the expected number of classes.
            - The loss is non-negative.
            - The model has parameters to train.
            - The test accuracy is greater than 0%.

    Example:
        >>> model = MobileNetV3SmallCNN()
        >>> training_params = TrainingParameters.from_json('training_params.json')
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> train_loader = your_train_loader
        >>> test_loader = your_test_loader
        >>> test_model_conformance_only(model, training_params, device, train_loader, test_loader)
    """
    # Ensure the model is on the correct device
    model.to(device)

    # Perform conformance tests
    test_model_conformance(model, train_loader, test_loader, device)
    print("Model passed all conformance checks!")


def validate_input_output(data, output):
    """
    Validates the shape and dimensions of the input data and the model output.

    Args:
        data (torch.Tensor): The input data tensor with shape (batch_size, channels, height, width).
        output (torch.Tensor): The output tensor from the model with shape (batch_size, num_classes).

    Raises:
        AssertionError: If the input or output tensors do not have the expected shapes or dimensions.
            - Input tensor should have 3 channels and be 4D.
            - Output tensor should have 10 classes and be 2D.

    Example:
        >>> data = torch.randn(32, 3, 224, 224)  # batch_size=32, 3 channels, 224x224 image size
        >>> output = torch.randn(32, 10)          # batch_size=32, 10 output classes
        >>> validate_input_output(data, output)
        # No exception raised, input and output are valid.
    """
    # Validate input shape
    assert (
        data.shape[1] == 3
    ), f"Input image should have 3 channels, found {data.shape[1]}."
    assert (
        data.dim() == 4
    ), f"Expected 4D input (batch_size, channels, height, width), found {data.dim()}."

    # Validate output shape
    assert (
        output.shape[1] == 10
    ), f"Output shape should be (batch_size, 10), found {output.shape}"
    assert (
        output.dim() == 2
    ), f"Expected 2D output (batch_size, num_classes), found {output.dim()}"
