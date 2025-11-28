<p align="center">
 <h1 align="center">QuickDraw - AI Drawing Recognition</h1>
</p>

[![GitHub stars](https://img.shields.io/github/stars/nughnguyen/quickdraw)](https://github.com/nughnguyen/quickdraw/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/nughnguyen/quickdraw?color=orange)](https://github.com/nughnguyen/quickdraw/network)
[![GitHub license](https://img.shields.io/github/license/nughnguyen/quickdraw)](https://github.com/nughnguyen/quickdraw/blob/master/LICENSE)

**Author:** nughnguyen

## Introduction

AI-powered drawing recognition based on Google's QuickDraw dataset. Features modern GUI with 50+ categories.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run GUI app
python gui_app.py
```

## Features

- 🎨 **Modern GUI** - Start/Stop buttons, no terminal commands needed
- 🎯 **50 Categories** - Airplane, apple, cat, dog, pizza, and more
- 📊 **Confidence Scores** - See prediction accuracy in real-time
- ⚙️ **Settings Panel** - Adjust color, threshold, display options

## Usage

### GUI Application (Recommended)

```bash
python gui_app.py
```

- Click **Start Camera** → Use colored object as pointer → **Start Drawing** → Draw → **Stop Drawing** to recognize

### Classic Camera App

```bash
python camera_app.py --color green --area 3000
```

- Press **SPACE** to start/stop drawing
- Press **Q** to quit

## Categories (50 total)

airplane, apple, basketball, bed, bicycle, bird, book, bowtie, cake, candle, car, cat, chair, circle, clock, cloud, computer, cup, dog, door, envelope, eyeglasses, fish, flower, guitar, hammer, hat, house, ice cream, key, leaf, lightning, moon, mountain, octopus, pandas, pants, pencil, pizza, rainbow, scissors, shoe, smiley face, snowflake, square, star, sun, t-shirt, tree, umbrella

## Training

```bash
# Download dataset
python download_data.py

# Train model
python train.py --total_images_per_class 10000 --num_epochs 20
```

## Requirements

- Python 3.6+
- OpenCV, PyTorch, NumPy, Pillow, scikit-learn

See `requirements.txt` for details.

## License

MIT License
