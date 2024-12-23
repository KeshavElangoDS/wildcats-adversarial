# Model Card

**Model:** MobileNetV3_small with AdamW optimizer

**Training Parameters:**
* batch_size: 64
* test_batch_size: 64
* epochs: 20
* learning rate: 1.0
* gamma: 0.5
* step_size: 5
* weight_decay: 0
* no_cuda: true
* no_mps: true
* dry_run: false
* seed: 1
* log_interval: 10
* save_model: true
* optimizer_type: adamw

**Model accuracy:**

*Before data augmentation:*

* Original test images : 94%
* Adversarial test images: 54%
* Defense method on original test image: 94%
* Defense method on adversarial test image: 50%

*After data augmentation:*

* Original test images : 94%
* Adversarial test images: 38%
* Defense method on original test image: 94%
* Defense method on adversarial test image: 50%

**Model training**:
* The model was trained using a batch size of 64, for 20 epochs with a learning rate of 1.0.
* The CrossEntropyLoss loss function was used for calculating the loss.
* We have a parameter called patience which determines when to stop the training provided the model does not improve after certain epochs.
* Seed is set to 1 for torch to produce reproducible results
* The number of epochs is determiend by the parameter patience.
* For every 10th batch the out is printed along with test accuracy for complete cycle.
* Optimizers aim to speed up training and improve the model's performance while preventing overfitting.
* **AdamW** is an optimizer adapts the learning rate for each parameter and includes weight decay for regularization.

**Limitations:**

* Overfitting on small datasets despite AdamW's regularization.
* Computational overhead from AdamW's momentum terms.
* Limited generalization on noisy data or tasks requiring fine-grained features.