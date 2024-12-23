[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/LU8t0ikG)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=17524211&assignment_repo_type=AssignmentRepo)
# RUC Practical AI - Final Project: Adversarial Attack and Defense Evaluation on WildCats Image Classification

## Purpose & Problem Addressed

This project aims to develop an efficient pipeline for training and testing deep learning models, such as MobileNetV3 and CNN architectures, on the **Wild Cats dataset**. The primary goal is to automate model training, validation, and evaluation processes, allowing users to quickly experiment with different models and configurations while ensuring robust testing through input-output validation and conformance checks.

The project also addresses a critical conservation issue: the monitoring and protection of endangered wildcat species. Wildcats face numerous threats, including habitat loss, illegal poaching, and human-wildcat conflict. Traditional tracking methods are often ineffective due to their elusive nature and the remoteness of their habitats. This project proposes the use of **camera trap images** to monitor wildcat populations, behavior, and health, providing valuable insights for conservation efforts. By leveraging deep learning models to analyze camera trap data, the project helps bridge gaps in monitoring, improves anti-poaching strategies, and supports the development of effective conservation plans for wildcat species.


### Key Features:
- Model training using optimizers like AdamW and Adadelta.
- Early stopping based on validation accuracy.
- Conformance testing and input-output validation to ensure model correctness.
- Integration with custom dataset handling for the Wild Cats dataset.
- Performance tracking (train and test loss/accuracy over epochs).

## Usage Instructions

### Cloning the Project
To get started with this project, you can clone it to your local machine by running the following command:

```bash
git clone  https://github.com/ruc-practical-ai/fall-2024-final-project-KeshavElangoDS.git
cd fall-2024-final-project-KeshavElangoDS
```

### Installing the Project
Ensure that you have Python 3.7+ and PyTorch installed. You can install the required dependencies using pip:

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

#### Poetry

This project is built on Python 3.12. Poetry is required for installation. To install Poetry, view the instructions [here](https://python-poetry.org/docs/).

#### Installing Python Dependencies Locally

To install locally, first install the required dependencies (Poetry), then clone the repository and navigate to its directory.

```bash
git clone  https://github.com/ruc-practical-ai/fall-2024-final-project-KeshavElangoDS.git
cd fall-2024-final-project-KeshavElangoDS
```

Configure Poetry to install its virtual environment inside the repository directory.

```bash
poetry config virtualenvs.in-project true
```

Install the repository's Python dependencies.

```bash
poetry install --no-root
```

Check where Poetry built the virtual environment with the following command.

```bash
poetry env info --path
```

Open the command pallette with `Ctrl` + `Shift` + `P` and type `Python: Select Interpreter`.

Now specify that VSCode should use the that interpreter (the one in `./.venv/Scripts/python.exe`). Once you specify this, Jupyter notebooks should show the project's interpreter as an option when you click the `kernel` icon or the small icon showing the current version of python (e.g., `Python 3.12.1`) and then click `Select Another Kernel`, and finally click `Python Environments...`.


## Running the Model
Once the dependencies are installed, you can start training a model using the provided pipeline. Below is an example of how to set up and run the training process with your data:

```bash
from finalproject.training_pipeline import train_modified_scheduler_with_early_stopping
from finalproject.nn_models import MobileNetV3SmallCNN
from finalproject.data_loading import read_wild_cats_annotation_file, WildCatsDataset

# Load dataset and prepare data loaders
annotations_file_path = 'path/to/annotations.csv'
train_data, test_data, validation_data, idx_to_class = read_wild_cats_annotation_file(annotations_file_path)

# Initialize dataset
train_loader = torch.utils.data.DataLoader(WildCatsDataset(img_dir='path/to/images', data=train_data), batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(WildCatsDataset(img_dir='path/to/images', data=test_data), batch_size=32)

# Model setup
model = MobileNetV3SmallCNN(req_grad=True, num_classes=10)

# Training configuration
training_params = {
    'lr': 0.001,
    'epochs': 20,
    'batch_size': 32,
    'test_batch_size': 32,
    'optimizer_type': 'adamw',
    'save_model': True,
    'dry_run': False,
    'no_cuda': False,
    'log_interval': 10,
    'step_size': 10,
    'gamma': 0.7,
    'seed': 1,
}

# Train the model
train_loss, test_loss, train_accuracies, test_accuracies, trained_model = train_modified_scheduler_with_early_stopping(
    model, training_params, 'cuda', train_loader, test_loader, 'wildcats_model'
)
```

## Performance Evaluation
After training, you can evaluate the model on the test dataset using the predict function to get the test loss and accuracy.

```bash
from finalproject.training_pipeline import predict

test_loss, test_accuracy = predict(trained_model, 'cuda', test_loader)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
```

## Known Issues

- **Dataset Format Assumptions**:  
  The code assumes that the dataset is in CSV format with specific column names (e.g., `class id`, `filepaths`, `labels`). If your dataset does not match this format, modifications will be needed in the `WildCatsDataset` class to properly load the data.

- **GPU Memory Usage**:  
  When training on large datasets or using complex models (e.g., MobileNetV3), GPU memory consumption might be high. If you encounter out-of-memory (OOM) errors, try reducing the batch size or enabling gradient accumulation.

- **MPS (Metal Performance Shaders) Support**:  
  The Metal Performance Shaders (MPS) backend for macOS is experimental. It may not work across all macOS configurations, and performance may vary. If you experience issues, try disabling MPS and default to CUDA or CPU.

- **Learning Rate Scheduling**:  
  The learning rate scheduler (`StepLR`) used in this project may require tuning for different model architectures. Depending on your task, the current configuration may not provide optimal results.

- **Early Stopping Behavior**:  
  The early stopping mechanism might stop training prematurely if there's no improvement in test accuracy for several epochs. This behavior might need to be fine-tuned depending on the dataset and model.

## Feature Roadmap

We plan to add the following features in future versions of this project:

1. **Advanced Data Augmentation**:  
   Add more data augmentation techniques, such as random cropping, rotation, color jitter, and flipping, to increase model robustness and generalization.

2. **Automatic Hyperparameter Tuning**:  
   Implement an automated hyperparameter optimization framework (e.g., Grid Search or Random Search) to explore different learning rates, batch sizes, and optimizer settings.

3. **Model Checkpoints and Resume Training**:  
   Add functionality to save model checkpoints periodically during training so that the training process can be resumed from the last checkpoint if interrupted.

4. **TensorBoard Integration**:  
   Integrate TensorBoard for better visualization of training and validation metrics such as loss, accuracy, and gradients.

5. **Multi-GPU Support**:  
   Enable multi-GPU training for faster model training on large datasets.

6. **Custom Metrics**:  
   Add support for custom evaluation metrics such as F1-score, Precision, Recall, etc., in addition to accuracy.

7. **Model Versioning**:  
   Implement model versioning with tools like DVC (Data Version Control) or MLflow to track experiments and results more efficiently.

## Contributing

We welcome contributions to improve this project. If you encounter bugs, have feature requests, or want to contribute enhancements, follow these steps:

### Steps to Contribute:
1. **Fork the Repository**:  
   Click on the "Fork" button in the top right corner of the repository page to create a copy of the project in your GitHub account.

2. **Clone the Repository**:  
   Clone the forked repository to your local machine using the following command:
   ```bash
   git clone  https://github.com/ruc-practical-ai/fall-2024-final-project-KeshavElangoDS.git
   cd wildcats-model
   ```
3. **Create a New Branch**  
   Create a new branch to work on your changes. It is recommended to use a descriptive name for your branch based on the feature or bug you're working on:
   ```bash
   git checkout -b feature-branch
   ```
4. **Make Changes**
    Make your desired changes to the code, whether that involves adding new features, fixing bugs, or improving documentation. Be sure to follow the project's coding standards and guidelines.

5. **Commit Your Changes**
    Once you've made the necessary changes, commit them to your branch. Provide a meaningful commit message describing the changes you've made:
    ```bash
    git commit -m "Add feature/fix bug"
    ```

6. **Push to Your Fork**
    Push your changes to your forked repository on GitHub:
    ```bash
    git push origin feature-branch
    ```
7. **Create a Pull Request (PR)**
    After pushing your changes to your forked repository, follow these steps to create a pull request:

    **Navigate to the GitHub Repository**  
    Go to the [main repository]( https://github.com/ruc-practical-ai/fall-2024-final-project-KeshavElangoDS.git) on GitHub where you want to contribute.

    **Click on "Compare & Pull Request"**  
    Once you're on the main page of your forked repository, you'll see a prompt to create a pull request. Click on the green **"Compare & Pull Request"** button.

    **Select Branches**  
    - In the **"base"** dropdown, select the branch you want to merge into, typically `main` (or `master`).
    - In the **"compare"** dropdown, select your feature branch (the branch you created earlier, e.g., `feature-branch`).

    **Describe the Changes**  
     In the PR description, provide a clear and concise explanation of what changes you made and why they were necessary. This helps the maintainers quickly understand the purpose of your PR.

    Example:
    ```markdown
    ### Summary of Changes:
    - Added a new data augmentation function for random cropping.
    - Fixed bug where `train.py` wasn't loading the dataset correctly for large batches.
    - Updated the model to include additional dropout layers for regularization.

    ### Related Issues:
    - Fixes #123 (Dataset loading bug)
    ```


## License
This repository is provided with an MIT license. See the `LICENSE` file.

## Contact
Please email Keshav Elango at ke270@scarletmail.rutgers.edu (academic) or keshavelangousa@gmail.com (personal) with questions, comments, bug reports, or suggestions for improvement.