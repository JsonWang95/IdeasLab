# Golf Ball Detection and Tracking

## Project Overview

This project implements a deep learning-based solution for detecting and tracking golf balls in video sequences. It utilizes a U-Net architecture with a pre-trained backbone for efficient and accurate localization of golf balls in various environments.

## Key Features

- Custom U-Net model with timm-based backbones for flexible architecture choices
- Separate datasets for detection and tracking tasks
- Data augmentation techniques to improve model generalization
- Heatmap-based approach for precise golf ball localization
- Training pipeline with learning rate scheduling and model checkpointing
- Visualization tools for model predictions and training progress

## Project Structure

- `src/model.py`: Contains the U-Net model architecture
- `src/dataset.py`: Implements custom datasets for golf ball detection and tracking
- `src/test_train.py`: Main training script with data loading, model training, and evaluation
- `src/timm_unet.py`: Defines the U-Net model using timm backbones
- `src/utils.py`: Utility functions for visualization and model checkpointing

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/golf-ball-detection-tracking.git
   cd golf-ball-detection-tracking
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train the model:
python src/test_train.py

This will start the training process, save model checkpoints, and generate visualizations in the `training_results_TIMESTAMP` directory.

## Data

The project expects data to be organized in the following structure:
data/
|-- detection/
|   |-- JPEGImages/
|   |-- Annotations/
|   `-- ImageSets/
`-- tracking/
    |-- JPEGImages/
    |-- Annotations/
    `-- ImageSets/

Ensure your data follows this structure before running the training script.

## Results

Training results, including loss plots and sample predictions, will be saved in the `training_results_TIMESTAMP` directory. You can monitor the training progress and model performance using these visualizations.
