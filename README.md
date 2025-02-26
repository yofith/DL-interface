# Interface

This is the interface for the AI model.

## Requirements

- Python 3.8 

## Installation

```bash
pip install flask flask-cors pillow torch torchvision opencv-python matplotlib numpy
```
## Run

```bash
python app.py
```
## Usage

```bash
curl -X POST -F "image=@path/to/your/image.jpg" http://localhost:5000/process_image
```

you can also use the interface to upload the image and get the result on the browser: 

```bash
http://localhost:5000/
```







