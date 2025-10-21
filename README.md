# PyTorch Learning Project

[![it](https://img.shields.io/badge/lang-it-red.svg)](README-it.md)

---

## About

This repository is a hands-on learning project for **PyTorch**, exploring the key steps in a deep learning workflow:

- Loading and working with datasets
- Creating DataLoaders
- Defining and training neural networks
- Saving models and performing inference

Everything in this project is inspired by and follows the **official PyTorch tutorials and guides**, adapted for step-by-step learning.

> ⚠️ Note: The project is actively updated. New tutorials, notebooks, and scripts may be added over time, so folder contents and names may expand.

---

## Project Structure

Current structure (subject to updates):

- `[00]_dataset_and_dataloader/` – Dataset & DataLoader exploration
- `[01]_model_creation_and_train/` – Neural network creation & training
- `[02]_model_loading_and_inference/` – Loading trained models & inference
- `[03]_first_exercise/` - Consolidates the previous three tutorials

> Each folder contains a **Jupyter Notebook** and a `main.py` script.  
> Datasets and saved models are gitignored (`data/` and `model/` folders) to keep the repository light.

---

## Installation

Recommended setup using **conda**:

### Create a new environment
```bash
conda create -n pytorch_course python=3.10
conda activate pytorch_course
```

### Install PyTorch with CUDA (if available)
Refer to the official guide here: https://pytorch.org/get-started

### Additional dependencies
```bash
pip install -r requirements.txt
```

---

## How to Use

1. Open Jupyter notebooks:


2. Follow the notebooks step by step:


3. Alternatively, run Python scripts directly:

```bash
python <script_name>.py
```

> Make sure your `data/` folder is present or let the scripts download the datasets automatically.

---

## Key Learning Points

- PyTorch **Dataset** and **DataLoader** API
- Creating **custom datasets**
- Building and training **feedforward neural networks**
- Using **loss functions** and **optimizers** (`SGD` / `Adam`)
- Evaluating models and **visualizing predictions**
- Saving and loading model weights for inference
- Building and training **convolutional neural networks**

---

## References

- PyTorch Tutorials: https://docs.pytorch.org/tutorials/beginner/basics/intro.html


---

## Contributing

This project is for learning and experimentation.  
Feel free to fork, experiment, and improve it!  
Since the project is actively updated, check back for new tutorials and examples.

---

