<p align="center">
 <h1 align="center">QuickDraw - AI Drawing Recognition</h1>
</p>

[![GitHub stars](https://img.shields.io/github/stars/nughnguyen/quickdraw)](https://github.com/nughnguyen/quickdraw/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/nughnguyen/quickdraw?color=orange)](https://github.com/nughnguyen/quickdraw/network)
[![GitHub license](https://img.shields.io/github/license/nughnguyen/quickdraw)](https://github.com/nughnguyen/quickdraw/blob/master/LICENSE)

**Author:** nughnguyen

## Introduction

An AI-powered drawing recognition application based on Google's QuickDraw dataset. This project features:

- **Modern GUI Application** - Easy-to-use interface with camera controls
- **Real-time Recognition** - Instant drawing recognition with confidence scores
- **50+ Categories** - Expanded dataset for richer experience
- **No Terminal Commands** - All controls available through GUI buttons

## Features

### 🎨 GUI Application (Recommended)

Run the modern GUI application with full controls:

```bash
python gui_app.py
```

**Features:**

- ▶️ Start/Stop camera with buttons (no more Ctrl+C!)
- ✏️ Toggle drawing mode
- 🎯 Real-time prediction with confidence scores
- ⚙️ Settings panel (color selection, area threshold)
- 🖼️ Optional canvas display
- 📊 Visual prediction display with icons

### 📷 Camera App (Classic)

Run the classic camera application:

```bash
python camera_app.py --color green --area 3000
```

**Controls:**

- Press **SPACE** to start/stop drawing
- Press **Q** to quit

**Requirements:**

- A pen or object with blue, red, or green color
- The object will be highlighted with a yellow circle

<p align="center">
  <img src="demo/quickdraw.gif" width=600><br/>
  <i>QuickDraw in action</i>
</p>

## Dataset

The dataset is from [Google's Quick Draw](https://console.cloud.google.com/storage/browser/quickdraw_dataset). This project uses **50 categories** for a richer recognition experience.

### Download Dataset

Automatically download all required dataset files:

```bash
python download_data.py
```

This will download .npy files for all 50 categories (~1-2GB total).

## Categories

The model recognizes **50 different categories**:

|          |            |            |           |          |
| -------- | :--------: | :--------: | :-------: | :------: |
| airplane |   apple    | basketball |    bed    | bicycle  |
| bird     |    book    |   bowtie   |   cake    |  candle  |
| car      |    cat     |   chair    |  circle   |  clock   |
| cloud    |  computer  |    cup     |    dog    |   door   |
| envelope | eyeglasses |    fish    |  flower   |  guitar  |
| hammer   |    hat     |   house    | ice cream |   key    |
| leaf     | lightning  |    moon    | mountain  | octopus  |
| panda    |   pants    |   pencil   |   pizza   | rainbow  |
| scissors |    shoe    | smile face | snowflake |  square  |
| star     |    sun     |  t-shirt   |   tree    | umbrella |

## Trained models

You could find my trained model at **trained_models/whole_model_quickdraw**

## Training

1. Download dataset files using the download script:

```bash
python download_data.py
```

2. Train the model:

```bash
python train.py --total_images_per_class 10000 --num_epochs 20 --batch_size 32
```

**Customize categories:** Edit `CLASSES` in `src/config.py` and re-download data.

**Training parameters:**

- `--optimizer`: sgd or adam (default: sgd)
- `--total_images_per_class`: Number of images per category (default: 10000)
- `--ratio`: Train/test split ratio (default: 0.8)
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Number of epochs (default: 20)
- `--lr`: Learning rate (default: 0.01)

## Experiments:

For each class, I take the first 10000 images, and then split them to training and test sets with ratio 8:2. The training/test loss/accuracy curves for the experiment are shown below:

<img src="demo/loss_accuracy_curves.png" width="800">

## Installation

1. Clone the repository:

```bash
git clone https://github.com/nughnguyen/quickdraw.git
cd quickdraw
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset:

```bash
python download_data.py
```

4. Run the GUI application:

```bash
python gui_app.py
```

## Requirements

- **Python 3.6+**
- **OpenCV (cv2)**
- **PyTorch**
- **NumPy**
- **Pillow**
- **scikit-learn**
- **TensorboardX**

See `requirements.txt` for specific versions.

## Project Structure

```
QuickDraw/
├── gui_app.py              # Modern GUI application
├── camera_app.py           # Classic camera application
├── train.py                # Model training script
├── download_data.py        # Dataset download script
├── requirements.txt        # Python dependencies
├── src/
│   ├── config.py          # Configuration and classes
│   ├── model.py           # Neural network model
│   ├── dataset.py         # Dataset loader
│   └── utils.py           # Utility functions
├── data/                   # Dataset files (.npy)
├── images/                 # Category icon images
├── trained_models/         # Saved models
└── demo/                   # Demo images/videos
```
