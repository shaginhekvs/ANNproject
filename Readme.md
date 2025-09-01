# ANNproject — Neural Network Mini-Projects (incl. ChatBot)

This repository contains a set of Jupyter notebooks for hands-on Artificial Neural Network (ANN) mini-projects. It includes a simple GPU check, a chatbot notebook, and additional experiments, with data and model checkpoints organized in dedicated folders. The project is notebook-driven, so you can run each experiment end-to-end from within Jupyter. ([GitHub][1])

---

## Repository Structure

```
.
├── checkpoints/                 # Saved model weights / artifacts
├── data/                        # Datasets or data placeholders
├── 281922.ipynb                 # Experiment notebook
├── gpu_test.ipynb               # Verify CUDA/GPU availability
├── miniproject1.ipynb           # Mini-project #1
├── miniproject2_ChatBot.ipynb   # ChatBot mini-project
├── miniproject3_2019.ipynb      # Mini-project #3
└── Readme.md                    # (this file)
```

> Tip: The chatbot lives in `miniproject2_ChatBot.ipynb`. ([GitHub][1])

---

## Quick Start

### 1) Clone

```bash
git clone https://github.com/shaginhekvs/ANNproject.git
cd ANNproject
```

### 2) Create an environment (choose one)

**Conda**

```bash
conda create -n annproj python=3.10 -y
conda activate annproj
```

**venv**

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

> The notebooks use standard PyData + PyTorch tooling. If you have a GPU, install a CUDA-enabled build of PyTorch that matches your system.

```bash
# PyTorch (CPU or pick a CUDA build from pytorch.org if you have a GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Common notebook deps
pip install jupyterlab ipykernel numpy pandas scikit-learn matplotlib
```

> If any notebook imports additional libraries, install them on demand with `pip install <package>`.

### 4) Launch Jupyter

```bash
jupyter lab
# or
jupyter notebook
```

Open any of the notebooks (e.g., `miniproject2_ChatBot.ipynb`) and run the cells top-to-bottom.

---

## GPU Check (Optional)

If you have an NVIDIA GPU and CUDA installed, verify it via the provided notebook:

* Open `gpu_test.ipynb` and run all cells, or run this in Python:

```python
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
```

---

## Data

Place datasets under `./data/`. Notebooks may expect specific file names or subfolders—check each notebook’s first cells for exact paths and preprocessing steps. ([GitHub][1])

---

## Checkpoints

Models and intermediate artifacts can be saved under `./checkpoints/`. Adjust save/load paths inside each notebook to match your workflow. ([GitHub][1])

---

## Notebook Index

* **`miniproject2_ChatBot.ipynb`** — Implements a simple ChatBot pipeline (data prep, model training/inference).
* **`miniproject1.ipynb`** — Mini-project #1 (see notebook header for task description).
* **`miniproject3_2019.ipynb`** — Mini-project #3 (see notebook header for task description).
* **`281922.ipynb`** — Additional experiment / exploration.
* **`gpu_test.ipynb`** — Environment sanity checks for GPU/CUDA.

> Each notebook contains its own narrative and cells for setup, training, and evaluation. ([GitHub][1])

---

## Suggested Workflow

1. Start with `gpu_test.ipynb` to confirm your environment.
2. Open a mini-project notebook and read the first markdown cell(s) for context and requirements.
3. Run the notebook sequentially; modify hyperparameters or data paths as needed.
4. Save outputs to `checkpoints/` or `data/processed/` for reuse across runs.

---

## Troubleshooting

* **CUDA not found / `torch.cuda.is_available() == False`**
  Install a CUDA-enabled PyTorch build that matches your NVIDIA driver + CUDA runtime (see the PyTorch Get Started page).
* **Missing package errors**
  `pip install <missing-package>` in your active environment.
* **Out-of-memory (OOM)**
  Reduce batch size or model size in the training cell; close other GPU apps.
* **Permission issues writing to `checkpoints/` or `data/`**
  Ensure the folders exist and you have write permissions.

---

## Contributing

* Keep notebooks tidy: clear extraneous state, checkpoint sparingly, and prefer relative paths.
* Add short markdown headers (Problem → Data → Model → Training → Results) to help others follow your work.
* If you add dependencies, note them near the top of the notebook.

---

## License

No license is currently specified in the repository. If you intend to share/modify this work, consider adding a license (e.g., MIT, Apache-2.0) at the repo root.

---

## Acknowledgements

Thanks to the authors and contributors of this repository. If you use part of the work (e.g., the ChatBot notebook), please credit the repo and link back to it. ([GitHub][1])

[1]: https://github.com/shaginhekvs/ANNproject "GitHub - shaginhekvs/ANNproject"
