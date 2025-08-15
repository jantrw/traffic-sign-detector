# Training Script for YOLOv8 on GTSRB Dataset #

### Prerequisites:
- Python 3.10 installed (do not use Python 3.12)
- A virtual environment with CUDA-enabled PyTorch (for example torch 2.8.0+cu126)
- NVIDIA RTX 3070 (or other supported GPUS) GPU with up-to-date drivers
- ultralytics YOLO package installed ('pip install ultralytics')
- Dataset organized in YOLO format with images, labels, and a traffic.yaml configuration file

### Setup instructions:
`git clone https://github.com/jantrw/traffic-sign-detector`
`cd traffic-sign-detector`

### Create and activate virtual environment:
`py -3.10 -m venv venv`
`venv\Scripts\activate`

### Install required packages:
`pip install ultralytics torch torchvision --index-url https://download.pytorch.org/whl/cu126`

### Script details:
Open scripts/train.ps1 and adjust parameters as needed:
  model          yolov8m.pt (or yolov8s.pt, yolov8n.pt)
  data           path to data/traffic.yaml
  imgsz          image size, for example 640 or 512
  epochs         number of training epochs, for example 50
  batch          batch size, use auto or specify number like 8
  device         GPU index, set value 0
  workers        number of CPU workers, for example 4 or 8
  optimizer      optimizer type, for example AdamW
  patience       early stopping patience, for example 20
  save           true or false to enable checkpoint saving
  plots          true or false to generate training plots
  project        directory for log files, default is runs
  tensorboard    true or false to enable TensorBoard logging (optional)

### Run the training:
`.\scripts\train.ps1`

### The script starts training on GPU if available. Training progress, including loss and metrics, is displayed in the console.

### Optional: Visualize training progress as image:
You can also **generate a summary image of the training progress while the model is still training**.
Run the following script after at least one epoch has completed:
`python scripts/train_model/convert_traindata_to_image.py`
This will analyze the `runs/detect/train/results.csv` file and create a single image containing two graphs showing the learning progress.
The image will be saved in the `scripts/train_model` directory, next to the training script and this utility script.

### Recommendations:
- If GPU load is too high, reduce imgsz from 640 to 512 (consistent with multiples of 32)
- Use smaller model like yolov8s.pt or yolov8n.pt for faster and less resource-intensive training
- Specify batch size to control GPU memory use, for example batch=8 or batch=4
- Perform a short test training run with epochs=10 to estimate runtime
- You can resume training from last checkpoint using resume=True

### License:
MIT License
