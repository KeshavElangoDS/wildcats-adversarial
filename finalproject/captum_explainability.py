"""
captum_explainabllity.py: Contains functions to provide explainablity for the model 

This module contains functions to perform Occlusion explainablity technique from Captum.
"""

from captum.attr import Occlusion
import torch
import matplotlib.pyplot as plt
import onnxruntime as ort

from captum.attr import visualization as viz
import numpy as np


def safe_normalize(image):
    """Safely normalize the image or attribution map to [0, 1]."""
    min_val, max_val = image.min(), image.max()
    if max_val == min_val:
        print(
            "Warning: The attribution map has no variance (min == max). \
              Returning the original image."
        )
        return image  # Avoid division by zero, return the image as is
    return (image - min_val) / (max_val - min_val)  # Normalizing between 0 and 1


def plot_explainability_with_captum_occlusion(
    model, input_image, target_class, overall_title, patch_size=8
):
    """
    Visualizes the explainability of a given model using occlusion-based attributions with Captum.

    This function computes attributions using the Occlusion method from Captum, which measures the
    importance of different regions in the input image by systematically occluding (masking)
    patches of the image and observing the effect on the model's output. The resulting attributions
    are then visualized alongside the original image using heatmaps to highlight areas with high
    attribution scores.

    Args:
        model (torch.nn.Module): The trained PyTorch model to explain.
        input_image (torch.Tensor): The input image tensor to be explained. Shape should be [C, H, W].
        target_class (int): The target class index for which the explanation is generated.
        overall_title (str): The title to display for the overall figure.
        patch_size (int, optional): The size of the occlusion patches. Default is 8.
            Larger values result in larger occlusion regions.

    Returns:
        None: The function visualizes the result using `matplotlib`. If visualization fails,
        an error message is printed.

    Raises:
        AssertionError: If there is an issue during visualization (e.g., if the attributions or
                        input image contain constant values).
    """
    model.eval()

    baseline = torch.zeros_like(input_image)

    occlusion = Occlusion(model)
    attributions_occ = occlusion.attribute(
        input_image,
        target=target_class,
        strides=(3, patch_size, patch_size),  # Modify as per your patch size
        sliding_window_shapes=(3, 15, 15),  # Modify for your image size
        baselines=baseline,
    )

    attributions_occ = attributions_occ.squeeze().cpu().detach().numpy()
    attributions_occ = safe_normalize(attributions_occ)

    # Normalize the input image to [0, 1] range and transpose to (H, W, C)
    original_image = input_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    original_image = safe_normalize(original_image)  # Normalize with safety check

    if original_image.shape[2] == 3:
        pass  # Correct format already (224, 224, 3)
    else:
        original_image = original_image.transpose(1, 2, 0)  # Transpose to (H, W, C)

    if attributions_occ.shape[2] == 3:
        pass
    else:
        attributions_occ = attributions_occ.transpose(1, 2, 0)

    if attributions_occ.ndim == 2:
        attributions_occ = attributions_occ[
            :, :, np.newaxis
        ]  # Add the channel dimension

    try:
        fig, _ = viz.visualize_image_attr_multiple(
            attributions_occ,
            original_image,
            ["original_image", "heat_map", "masked_image"],
            ["all", "positive", "positive"],  # Show positive attribution only
            show_colorbar=True,
            titles=[
                "Original Image",
                "Positive Attribution",
                "Masked Positive Attribution",
            ],
            fig_size=(8, 6),
        )

        fig.suptitle(
            f"{overall_title} - Occlusion Based Attribution Visualization", fontsize=16
        )

    except AssertionError as e:
        print(f"Error during visualization: {e}")
        print("Possible cause: Attribution map or image might have constant values.")
        return


# Load the ONNX model using onnxruntime
class ONNXModelWrapper(torch.nn.Module):
    def __init__(self, onnx_model_path):
        super().__init__()
        self.session = ort.InferenceSession(onnx_model_path)

    def forward(self, input_tensor):
        input_tensor = input_tensor.detach()

        input_array = input_tensor.cpu().numpy()

        # Run inference using onnxruntime
        inputs = {self.session.get_inputs()[0].name: input_array}
        output = self.session.run(None, inputs)

        return torch.tensor(output[0])
