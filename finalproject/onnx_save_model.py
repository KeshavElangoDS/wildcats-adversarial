"""
onnx_save_model.py: Saves the trained model with given name

This module contains function to save trained model with provided model name.
"""

import os
from pathlib import Path
import torch


def export_model_to_onnx(model, model_name):
    """
    Export a trained PyTorch model to the ONNX format.

    Args:
        model (torch.nn.Module): The trained PyTorch model to be exported.
        model_name (str): The name for the exported ONNX model file (without the file extension).

    Returns:
        str: The file path of the saved ONNX model.

    Example:
        export_model_to_onnx(mobilenet_trained_model, "mobilenet_trained_model")

    Notes:
        - This function saves the model in the `./finalproject/models/` directory relative
        to the parent of the current working directory.
        - The function uses `batch_size=64` as a placeholder input for exporting the model.
        - The ONNX model will be saved with a `.onnx` extension.
    """
    # Get the current working directory and set the final project path
    curr_work_dir = Path.cwd()
    current_dir_abs = os.path.abspath(curr_work_dir)
    parent_dir = os.path.dirname(current_dir_abs)
    finalproject_path = os.path.join(parent_dir, "./finalproject")

    model.eval()

    batch_size = 64
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    torch_out = model(x)

    onnx_file_path = f"{finalproject_path}/models/{model_name}.onnx"

    torch.onnx.export(
        model,
        x,
        onnx_file_path,
        export_params=True,  # Store trained parameter weights inside the model file
        opset_version=10,  # ONNX version to export the model to
        do_constant_folding=True,  # Execute constant folding for optimization
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},  # Variable length axes
            "output": {0: "batch_size"},
        },
    )

    print(f"Model has been saved to: {onnx_file_path}")
    return onnx_file_path
