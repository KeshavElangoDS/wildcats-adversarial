"""
adversarial_attack_defense.py: Performs the adversarial attack and defense on model

This module contains the functions to perform FGSM(Fast Gradient Sign Method), 
displays the pertubed images and the absolute difference between original and benign images.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from torch.utils.data import DataLoader
from art.defences.preprocessor import JpegCompression, SpatialSmoothing


def preprocess_image(image):
    """
    Preprocesses an image for model input.

    Args:
        image: A PIL image or a numpy array representing the image.

    Returns:
        A normalized PyTorch tensor.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Convert to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if isinstance(image, Image.Image):
        image = transform(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        image = transform(image)
    return image


def show_images(
    images_array, sqrt_n_images, predicted_labels, original_labels, title=""
):
    """
    Display images in a grid format with predicted and original labels above each image.

    Args:
        images_array: The array of images to display.
        sqrt_n_images: The square root of the number of images to be displayed (for grid size).
        predicted_labels: The predicted labels for each image (numeric indices).
        original_labels: The original (ground truth) labels for each image (numeric indices).
        title: The title to be displayed at the top of the image grid.
    """
    # Mapping indices to class names
    idx_to_class = {
        0: "AFRICAN LEOPARD",
        1: "CARACAL",
        2: "CHEETAH",
        3: "CLOUDED LEOPARD",
        4: "JAGUAR",
        5: "LIONS",
        6: "OCELOT",
        7: "PUMA",
        8: "SNOW LEOPARD",
        9: "TIGER",
    }

    fig, axes = plt.subplots(sqrt_n_images, sqrt_n_images, figsize=(8, 8))
    fig.suptitle(title, fontsize=16)  # Title for the entire grid

    for idx, ax in enumerate(axes.flat):
        # Transpose the image to (H, W, C) format for displaying
        img = images_array[idx]

        # If the image is a 3D array (C, H, W), transpose it to (H, W, C)
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))

        ax.imshow(img)
        ax.axis("off")

        # Convert numeric labels to strings
        original_label = idx_to_class[original_labels[idx]]
        predicted_label = idx_to_class[predicted_labels[idx]]
        color = "green" if original_label == predicted_label else "red"

        ax.text(
            0.5,
            -0.1,
            f"Prediction: {predicted_label}\nOriginal: {original_label}",
            color=color,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(
                facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.3"
            ),
        )

    plt.show()


def show_delta_images(
    benign_images, perturbed_images, sqrt_n_images, predicted_labels, original_labels
):
    """
    Display delta images (absolute difference between benign and adversarial images) in a grid format.

    Args:
        benign_images: The original (benign) images.
        perturbed_images: The adversarial (perturbed) images.
        sqrt_n_images: The square root of the number of images to be displayed.
        predicted_labels: The predicted labels for each image.
        original_labels: The original (ground truth) labels for each image.
    """
    delta_images = np.abs(
        benign_images - perturbed_images
    )  # Compute the delta (absolute difference)
    delta_images = np.clip(delta_images, 0, 1)

    max_delta = np.max(delta_images)
    if max_delta > 1:
        delta_images /= max_delta

    print("Displaying Delta Images (Perturbations)")
    show_images(
        delta_images,
        sqrt_n_images,
        predicted_labels,
        original_labels,
        title="Delta Images (Perturbations)",
    )


def test_fgsm_attack_batch(
    onnx_model_wrapper, pytorch_model, test_dataloader, epsilon=0.1, sqrt_n_images=3
):
    """
    This function accepts an ONNX model, iterates over the test_dataloader,
    and performs FGSM attack on each image in the batch using ART.
    Displays a subset of perturbed images and delta images (perturbations).
    Also calculates and prints the accuracy for both the original and perturbed images.

    Args:
        onnx_model_wrapper: The ONNXModelWrapper for the model.
        pytorch_model: The PyTorch model (used for computing gradients).
        test_dataloader: The DataLoader that provides test images and labels.
        epsilon: The perturbation magnitude (default=0.1).
        sqrt_n_images: The square root of the number of images to display.

    Returns:
        None (prints results and shows images).
    """
    # Wrap PyTorch model with ART's PyTorchClassifier
    classifier = PyTorchClassifier(
        model=pytorch_model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW(pytorch_model.parameters()),
        input_shape=(3, 224, 224),
        nb_classes=10,
    )

    attack = FastGradientMethod(estimator=classifier, eps=epsilon)

    total_images = 0
    correct_original = 0
    correct_perturbed = 0

    for batch_idx, batch in enumerate(test_dataloader):
        images = batch["image_input"]
        labels = batch["target"]  # Batch of labels

        labels = torch.tensor(labels).long()

        images = torch.stack([preprocess_image(image) for image in images]).to(
            torch.float32
        )
        images = (
            images.numpy()
        )  # Convert to numpy array for ART (expected input format)

        perturbed_images = attack.generate(x=images)

        original_preds = torch.argmax(torch.tensor(classifier.predict(images)), axis=1)
        perturbed_preds = torch.argmax(
            torch.tensor(classifier.predict(perturbed_images)), axis=1
        )

        correct_original += (original_preds == torch.tensor(labels)).sum()
        correct_perturbed += (perturbed_preds == torch.tensor(labels)).sum()
        total_images += len(labels)

        # Convert predictions to numpy for easy access in show_images
        predicted_labels = perturbed_preds.numpy()
        original_labels = labels.numpy()

        n_images = sqrt_n_images**2
        batch_size = perturbed_images.shape[0]
        random_choice = np.random.choice(batch_size, n_images, replace=False)

        plot_images_adv = perturbed_images[random_choice]
        plot_images_adv = np.clip(plot_images_adv, 0, 1)

        print("Displaying Adversarial Images")
        show_images(
            plot_images_adv,
            sqrt_n_images,
            predicted_labels[random_choice],
            original_labels[random_choice],
            title="Adversarial Images (Perturbed)",
        )

        plot_images_benign = images[random_choice]
        plot_images_benign = np.clip(plot_images_benign, 0, 1)
        show_delta_images(
            plot_images_benign,
            plot_images_adv,
            sqrt_n_images,
            predicted_labels[random_choice],
            original_labels[random_choice],
        )

    accuracy_original = (correct_original / total_images) * 100
    accuracy_perturbed = (correct_perturbed / total_images) * 100

    print(f"\nOverall Accuracy on Original Images: {accuracy_original:.2f}%")
    print(f"Overall Accuracy on Adversarial Images: {accuracy_perturbed:.2f}%")


def apply_fgsm_to_model(
    onnx_model_wrapper, pytorch_model, test_dataloader, epsilon=0.1
):
    """
    Applies the FGSM attack to the specified ONNX model on the entire test dataset using ART.

    Args:
        onnx_model_wrapper: The ONNX model wrapped with ONNXModelWrapper.
        pytorch_model: The PyTorch model (used for gradient computation).
        test_dataloader: The DataLoader that provides test images and labels.
        epsilon: The perturbation magnitude (default=0.1).
    """
    # Perform FGSM attack on the entire dataset using ART
    test_fgsm_attack_batch(onnx_model_wrapper, pytorch_model, test_dataloader, epsilon)


def unnormalize_image(image, mean, std):
    """
    Unnormalizes the image by multiplying by the standard deviation and adding the mean for each channel.

    Args:
        image (numpy.ndarray or torch.Tensor): The image to unnormalize.
        mean (list or tuple): The mean used for normalization.
        std (list or tuple): The standard deviation used for normalization.

    Returns:
        numpy.ndarray: The unnormalized image.
    """
    # Unnormalize
    for c in range(image.shape[1]):  # Assuming image is in the format (N, C, H, W)
        image[:, c, :, :] = image[:, c, :, :] * std[c] + mean[c]

    # Clip the values to [0, 1] to ensure valid pixel range for further processing
    image = np.clip(image, 0, 1)
    return image


def test_fgsm_attack_batch_with_defense(
    onnx_model_wrapper,
    pytorch_model,
    test_dataloader,
    epsilon=0.1,
    sqrt_n_images=3,
    jpg_quality=50,
    window_size=3,
):
    """
    This function applies defenses (e.g., JPEG Compression, Spatial Smoothing) to adversarial
    images and evaluates the model's performance on both original and adversarial examples
    after applying the defenses.

    Args:
        onnx_model_wrapper: The ONNXModelWrapper for the model.
        pytorch_model: The PyTorch model (used for computing gradients).
        test_dataloader: The DataLoader that provides test images and labels.
        epsilon: The perturbation magnitude (default=0.1).
        sqrt_n_images: The square root of the number of images to display.

    Returns:
        None (prints results and shows images).
    """
    # Wrap PyTorch model with ART's PyTorchClassifier
    classifier = PyTorchClassifier(
        model=pytorch_model,
        loss=F.cross_entropy,
        optimizer=torch.optim.Adadelta(pytorch_model.parameters()),
        input_shape=(3, 224, 224),
        nb_classes=10,
    )

    attack = FastGradientMethod(estimator=classifier, eps=epsilon)

    # Apply JPEG compression and Spatial smoothing as defenses
    clip_values = (0.0, 1.0)
    jpeg_compression = JpegCompression(quality=jpg_quality, clip_values=clip_values)
    spatial_smoothing = SpatialSmoothing(window_size=window_size)

    total_images = 0
    correct_original = 0
    correct_perturbed = 0

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for _, batch in enumerate(test_dataloader):
        images = batch["image_input"]
        labels = batch["target"]

        labels = torch.tensor(labels).long()

        images = torch.stack([preprocess_image(image) for image in images]).to(
            torch.float32
        )
        images = (
            images.numpy()
        )  # Convert to numpy array for ART (expected input format)
        perturbed_images = attack.generate(x=images)

        # Unnormalize the images and adversarial examples
        images_unnormalized = unnormalize_image(perturbed_images.copy(), mean, std)

        # Apply defense: JPEG compression and spatial smoothing on adversarial images
        perturbed_images_defended = jpeg_compression(images_unnormalized)

        # If jpeg_compression returns a tuple, extract only the image part
        if isinstance(perturbed_images_defended, tuple):
            perturbed_images_defended = perturbed_images_defended[0]

        perturbed_images_defended = spatial_smoothing(perturbed_images_defended)

        if isinstance(perturbed_images_defended, tuple):
            perturbed_images_defended = perturbed_images_defended[0]

        perturbed_images_defended = np.array(perturbed_images_defended)

        if (
            len(perturbed_images_defended.shape) != 4
            or perturbed_images_defended.shape[1] != 3
        ):
            print(
                f"Unexpected shape for perturbed images: {perturbed_images_defended.shape}"
            )
            continue

        # Get predictions for original and defended adversarial images
        original_preds = torch.argmax(torch.tensor(classifier.predict(images)), axis=1)
        perturbed_preds = torch.argmax(
            torch.tensor(classifier.predict(perturbed_images_defended)), axis=1
        )

        correct_original += (original_preds == torch.tensor(labels)).sum()
        correct_perturbed += (perturbed_preds == torch.tensor(labels)).sum()
        total_images += len(labels)

        # Convert predictions to numpy for easy access in show_images
        predicted_labels = perturbed_preds.numpy()
        original_labels = labels.numpy()

        n_images = sqrt_n_images**2
        batch_size = perturbed_images.shape[0]
        random_choice = np.random.choice(batch_size, n_images, replace=False)

        plot_images_adv = perturbed_images_defended[random_choice]
        plot_images_adv = np.clip(plot_images_adv, 0, 1)

        print("Displaying Adversarial Images after Defense")
        show_images(
            plot_images_adv,
            sqrt_n_images,
            predicted_labels[random_choice],
            original_labels[random_choice],
            title="Adversarial Images (After Defense)",
        )

        plot_images_benign = images[random_choice]
        plot_images_benign = np.clip(plot_images_benign, 0, 1)
        show_delta_images(
            plot_images_benign,
            plot_images_adv,
            sqrt_n_images,
            predicted_labels[random_choice],
            original_labels[random_choice],
        )

    accuracy_original = (correct_original / total_images) * 100
    accuracy_perturbed = (correct_perturbed / total_images) * 100

    print(f"\nOverall Accuracy on Original Images: {accuracy_original:.2f}%")
    print(f"Overall Accuracy on Defended Adversarial Images: {accuracy_perturbed:.2f}%")
