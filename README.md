# Vocal Health Detection Using Audio Classification

This project is a deep learning pipeline designed to classify vocal health conditions—such as *Laryngozele*, *Normal*, and *Vox senilis*—from audio recordings of vocalizations. Using a Long Short-Term Memory (LSTM) neural network, the project identifies patterns in audio features to aid in diagnosing vocal health conditions based on vocal characteristics.

---

## Dataset Information

The dataset used for this project contains vocal audio recordings of both healthy individuals and patients with vocal conditions. The conditions represented are:
- **Laryngozele**
- **Vox senilis**
- **Normal**

These audio files are sourced from the [Saarbrücken Voice Database](http://stimmdb.coli.uni-saarland.de/), which hosts a wide variety of speech samples aimed at supporting voice and speech research.

## Project Structure

The core structure of this project includes components for data loading, augmentation, model building, training, evaluation, and hyperparameter tuning.

---

### 1. Data Loading and Preprocessing

Audio data is loaded from a specified directory where subdirectories are organized by vocal health condition. Each `.wav` file is:
- Converted to a Mel spectrogram—a widely used audio feature representation in deep learning tasks.
- Standardized in length by padding or truncating spectrograms to ensure uniform input dimensions for the model.

---

### 2. Data Augmentation

To increase data variability and model robustness, the project uses an **AudioAugmentation** class that:
- Applies random volume adjustment.
- Adds Gaussian noise to audio samples with a configurable noise level.

This approach improves model generalization by simulating real-world variations in audio data.

---

### 3. Model Architecture

The model is a **Long Short-Term Memory (LSTM) neural network** with:
- **Input layer** to receive Mel spectrogram features.
- **LSTM layer** for capturing sequential dependencies in audio signals.
- **Dropout layer** for regularization, which deactivates neurons randomly to prevent overfitting.
- **Fully connected output layer** to classify the input into one of the three vocal health conditions.

---

### 4. Hyperparameter Tuning

Hyperparameters (e.g., batch size, learning rate, hidden layer size, dropout rate) are tuned to find the best configuration for optimal validation accuracy. **L2 regularization** is applied to further reduce overfitting.

---

### 5. Training and Validation

- The model is trained for a specified number of epochs, using **cross-entropy loss** as the objective function.
- **Adam optimizer** is used for efficient gradient-based optimization.
- **Training and validation accuracies** are recorded for each epoch, enabling performance tracking and convergence assessment.

---

### 6. Evaluation and Visualization

The project includes:
- **Accuracy and Loss Plotting**: Displays training loss and accuracies alongside validation accuracies across epochs, making it easy to evaluate model performance visually.
- **Best Model and Hyperparameters**: After training, the best model configuration (based on validation accuracy) is reported along with the corresponding hyperparameters.

---

## Results and Hyperparameters

The project outputs:
- **Best Validation Accuracy**: Achieved on the test set.
- **Optimal Hyperparameters**: Configuration of hyperparameters that yielded the best validation performance.

---

## Code Setup and Execution

1. **Dependencies**:
   - `torch`
   - `torchaudio`
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `sklearn`

2. **Execution**:
   - Initialize the dataset by setting the `root_dir` variable to point to the local dataset directory.
   - Instantiate the `VocalDataset` class with audio augmentation enabled.
   - Train the model using the provided hyperparameter grid search loop to find the best model configuration.

3. **Training Example**:
   ```python
   vocal_dataset = VocalDataset(root_dir=root_dir, augment=AudioAugmentation())
   ```

## Project Outcome

This project demonstrates the feasibility of detecting vocal health conditions from audio data, supporting potential applications in medical audio diagnostics. The model’s accuracy and generalizability may be further enhanced by exploring additional audio data and incorporating more advanced neural architectures.

