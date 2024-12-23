"""
data_loading: contains functions to load the wild cats data

This module contains functions to load a custom dataset, and display image in batches, 
plot train and test loss of neural network models 
"""

from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2


class WildCatsDataset(Dataset):

    def __init__(self, img_dir, data, transform=None):
        self.img_dir = img_dir
        self.data = data.reset_index(drop=True)
        self.transform = transform
        self.preprocess_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        class_id = self.data.loc[idx, "class id"]
        filepath = self.data.loc[idx, "filepaths"]
        label = self.data.loc[idx, "labels"]

        img = read_image(os.path.join(self.img_dir, filepath))

        if self.transform:
            img = self.transform(img)

        image_input = self.preprocess_transform(img)

        item_dict = {
            "target": class_id,
            "image_input": image_input,
            "image": img,
            "label": label,
        }

        return item_dict


def read_wild_cats_annotation_file(annotations_file_path):
    """
    Reads the Wild Cats dataset annotations file and splits the data into training, testing,
    and validation sets. Also maps the class IDs to their corresponding class labels.

    Args:
        annotations_file_path (str): Relative path to the annotations CSV file. The file should
                                      contain metadata about the Wild Cats dataset, including 
                                      dataset splits, class IDs, and labels.

    Returns:
        tuple: A tuple containing the following elements:
            - train_data (pd.DataFrame): A DataFrame containing training data.
            - test_data (pd.DataFrame): A DataFrame containing testing data.
            - validation_data (pd.DataFrame): A DataFrame containing validation data.
            - idx_to_class (dict): A dictionary mapping class IDs to their corresponding 
            class labels.
        
    Example:
        train_data, test_data, validation_data, idx_to_class = \
            read_wild_cats_annotation_file('path/to/annotations.csv')
        print(train_data.head())  # Display the first few rows of the training data
        print(idx_to_class)  # Show the class ID to class label mapping
    """
    curr_work_dir = Path.cwd()
    current_dir_abs = os.path.abspath(curr_work_dir)
    parent_dir = os.path.dirname(current_dir_abs)
    wildcats_df = pd.read_csv(os.path.join(parent_dir, annotations_file_path))

    train_data = wildcats_df[wildcats_df["data set"] == "train"]
    test_data = wildcats_df[wildcats_df["data set"] == "test"]
    validation_data = wildcats_df[wildcats_df["data set"] == "valid"]

    idx_to_class = {
        int(key): val
        for key, val in zip(
            list(wildcats_df["class id"].unique()), list(wildcats_df["labels"].unique())
        )
    }

    return train_data, test_data, validation_data, idx_to_class


def plot_batch_images(batch, rows, columns):
    """
    Function to plot a grid of images from a batch.
    
    Parameters:
    - batch: A dictionary containing the images and labels. 
             batch["image"] is expected to be a tensor with shape \
                (batch_size, channels, height, width).
             batch["label"] is expected to be a list or tensor of labels\
                  corresponding to each image.
    - rows: Number of rows in the image grid.
    - columns: Number of columns in the image grid.
    """
    figure = plt.figure(figsize=(6, 6))

    for index in range(1, rows * columns + 1):
        if index <= len(batch["image"]):
            image = batch["image"][index - 1]
            label = batch["label"][index - 1]

            figure.add_subplot(rows, columns, index)
            plt.axis("off")
            plt.imshow(
                image.permute(1, 2, 0).squeeze()
            )  # Convert channels-first to channels-last format
            plt.title(label)

    plt.tight_layout()
    plt.show()


def plot_train_test_loss(epochs, train_loss_lst, test_loss_lst, model_title):
    """
    Plots the training and test loss curves over the course of training.

    This function generates a plot to visualize how the training and test loss change
    across epochs. It helps in analyzing the model's performance during training and
    can reveal patterns such as overfitting or underfitting.

    Args:
        epochs (list or np.array): List or array of epoch numbers (e.g., [1, 2, 3, ..., N]).
        train_loss_lst (list or np.array): List or array of training loss values recorded for each epoch.
        test_loss_lst (list or np.array): List or array of test loss values recorded for each epoch.
        model_title (str): The title of the model, which will be included in the plot title.

    Returns:
        None: This function does not return any values. It generates and displays a plot.

    Raises:
        ValueError: If the lengths of `train_loss_lst`, `test_loss_lst`, and `epochs` do not match.
    """
    plt.style.use("classic")

    fig, ax1 = plt.subplots(1, figsize=(8, 6))

    ax1.plot(epochs, train_loss_lst, "o-", linewidth=1.5, label="Train Loss")
    ax1.plot(epochs, test_loss_lst, "o-", linewidth=1.5, label="Test Loss")
    ax1.set_title(f"Epochs vs Loss - {model_title}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="best")
    ax1.set_xlim(0.0, 23.0)
    ax1.grid()

    plt.show()
