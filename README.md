# midi-velocity-colorizer
PyTorch implementation for filling MIDI velocities from given MIDI notes. The model is an U-Net image colorizor &amp; trained on expert piano performances from the Piano-e-Competition (MAESTRO dataset).

This repository provides supplementary materials for our paper:
**"MIDI Velocity Prediction using U-Net Image Colorizer"** 

---

## 📁 Code Contents
- `train.py` — model training entry point
- `evaluation.ipynb` — reproduce Tables 2 & 3 quantitative results in our paper
- `interface.ipynb` — colorize & visualize the midi; reproduce Figures 1–4 in our paper
- `subjective_test_audio.tar` — listening test audio samples, generated using our colorized MIDI via PianoTeq 8. For details, see "Section 5.2" of our paper.
- `compare/` — reference models: Flat and Kim2023's Seq2Seq model.

## 📝 Additional Materials
- Paper: current folder
- WandB workspace: https://wandb.ai/zhanh-uwa/2025_cmmr?nw=nwuserzhanh
- WandB report: https://api.wandb.ai/links/zhanh-uwa/wpzvcb76
  - This report includes quantitative results (Tables 2 and 3) and a hyperparameter search not included in the main paper.
- The code will be further cleaned and finalized if the paper is accepted.
  - A Google Colab notebook in progress to help users process MIDI files without local code running environment.

---

## 0. Hyperparameter Setup

All training settings are defined in `conf/config.yaml`.  
- Some data filtering operations were implemented but not used in our experiments.
- To reproduce our results, please refer to the training logs and parameter tracking in our WandB workspace.

---

## 1. Dataset Setup

Please download the following datasets and place them under the `/dataset` folder:
- [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro)
- [Saarland Music Data v2](https://zenodo.org/records/13753319)

---

## 2. Environment Setup

Tested on:
- Ubuntu 20.04 (CUDA 12.0)
- Ubuntu 22.04 (CUDA 12.2)

### Using Conda & Pre-built Env (recommended):
```bash
conda env create -f environment.yaml
```

### Manual Build Your Env:
```bash
conda create --name velocity_pred python=3.11
conda install pytorch=2.2.2 torchvision=0.17.2 torchaudio=2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install lightning -c conda-forge
```

---

## 3. WandB Integration (optional)

We use [Weights & Biases](https://wandb.ai) for experiment tracking.

```bash
wandb login
```

If preferred, you can switch to TensorBoard by modifying `train.py`.

---

## 4. Train the Model (Ablation Study – Table 4)

Edit training options in `conf/config.yaml`. Then run:

```bash
# Re-implemented ConvAE baseline
python train.py exp.test_dataset="MAESTRO" matrix.seg_time=10 ae.model="ConvAE" loss.type="BCELoss" loss.mask='element_wise' loss.weight='u_shape' loss.cosim=0.2 exp.save_k_ckpt=10

# Proposed U-Net with 2x2 attention window
python train.py exp.test_dataset="MAESTRO" matrix.seg_time=10 ae.model="UNet" ae.ablation="attn" ae.attn_window=2 loss.type="BCELoss" loss.mask='element_wise' loss.weight='u_shape' loss.cosim=0.2 exp.save_k_ckpt=10
```

---

## 5. Evaluate the Model (Tables 2 & 3)

Use `evaluation.ipynb` to reproduce the quantitative results using saved checkpoints in `results/checkpoints`.

- For reference models:
  - Flat model: `compare/Flat_model/flat_evaluation.ipynb`
  - Kim2023 Seq2Seq model: `compare/Kim2023_model/seq2seq_evalaution.ipynb`

---

## 6. Interface & Demo (Figures 1–4)

Use `interface.ipynb` to visualize results or interact with trained models.  
The best U-Net and ConvAE models are pre-loaded for convenience.

---
