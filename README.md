# 🎹 MusicAI — Mood-Based Classical Music Generator

> Generate original piano compositions using deep learning.  
> A **2-layer stacked LSTM with Bahdanau Attention** trained on **501 classical MIDI files from Bach, Beethoven and Chopin**.

Select a mood → generate a piano composition → listen instantly.

Built entirely with **PyTorch** (no GenAI APIs).

---

# 🎬 Live Demo

👉 https://huggingface.co/spaces/rahateatharva/MusicLSTM

---

# 🖥️ Interface

<img width="1918" height="868" src="https://github.com/user-attachments/assets/134dde77-8b18-432a-99b2-33e94160829d">

<img width="1919" height="875" src="https://github.com/user-attachments/assets/353925e4-10b5-4b13-90ea-4a1f95435953">

Users can:

- Select a **mood**
- Adjust **creativity (temperature)**
- Control **composition length**
- Generate and **play piano music instantly**

---

# 🎯 What The System Does

The model generates **new piano compositions conditioned on mood**.

| Mood | Key | BPM | Composer Influence |
|------|-----|-----|-------------------|
| 🌿 Calm | Major | 76 | Bach |
| 🌧️ Sad | Minor | 60 | Chopin |
| 🌹 Romantic | Major | 72 | Beethoven |
| ⚡ Energetic | Major | 140 | Beethoven |
| 🌑 Dark | Minor | 88 | Bach |

---

# 🏗 Model Architecture

```
Input Sequence (seq_len = 64 tokens)
        ↓
Embedding (577 → 64)
        ↓
LSTM Layer 1 (hidden = 512, dropout = 0.3)
        ↓
LSTM Layer 2 (hidden = 512)
        ↓
Bahdanau Attention
        ↓
Linear Layer (512 → 577)
        ↓
Softmax + Temperature Sampling
        ↓
Generated Token Sequence
```

**Total Parameters**

```
3,875,441
```

---

# 🎼 Token Representation

The model uses a **combined token format**:

```
{pitch}_{duration}
```

Examples:

```
C4_0.5
G#3_1.0
E4_0.25
```

This allows the model to learn **melodic and rhythmic patterns jointly**.

Vocabulary size:

```
577 tokens
```

---

# 📊 Model Comparison (Ablation Study)

All models were trained using **identical hyperparameters and dataset**.

| Model | Best Loss | Best Accuracy | Notes |
|------|------|------|------|
| Vanilla RNN | 5.43 | 1.31% | Vanishing gradient |
| GRU | 3.79 | 15.82% | Overfits early |
| Transformer | 3.89 | 13.60% | Dataset too small |
| **LSTM + Attention** | **1.89** | **~37%** | Best model |

---

## Why LSTM Beat Transformer

The dataset contains only **501 MIDI files (~375k sequences)**.

Transformers usually require **very large datasets** to train well.

LSTMs have an **inductive bias toward sequential patterns**, making them more effective for smaller music datasets.

---

## Why Vanilla RNN Failed

Vanilla RNN suffers from **vanishing gradients** for longer sequences.

With `seq_len = 64`, gradients collapse and the model fails to learn meaningful patterns.

LSTM gates regulate gradient flow and solve this problem.

---

# 🗂 Dataset

| Composer | Files |
|----------|-------|
| Bach | 410 |
| Beethoven | 30 |
| Chopin | 61 |

Total dataset:

```
501 MIDI files
375,552 tokens
577 unique tokens
```

---

# ⚙️ Data Processing Pipeline

1. Parse MIDI files using `music21`
2. Extract note + duration pairs
3. Normalize durations
4. Simplify chords
5. Build vocabulary
6. Create sliding window sequences

Duration normalization:

```
Round to nearest 0.25
Clamp to range [0.25, 2.0]
```

Chord preprocessing:

```
Keep lowest pitch
```

Sequence generation:

```
Sequence length = 64
Stride = 1
```

---

# 🎼 Generation Pipeline

```
Seed Sequence (64 tokens)
        ↓
Autoregressive LSTM inference
        ↓
Temperature sampling
        ↓
Mood-based scale filtering
        ↓
Chord injection every 4 notes
        ↓
Velocity dynamics
        ↓
Humanized durations (±5%)
        ↓
music21 → MIDI
        ↓
FluidSynth → WAV
```

---

# 🚀 Training

Training performed on **NVIDIA RTX 4050**.

| Stage | Epochs | Learning Rate | Loss |
|------|------|------|------|
| Initial Training | 50 | 0.001 | 2.52 |
| Finetune 1 | 20 | 0.0001 | 2.18 |
| Finetune 2 | 20 | 0.0001 | 2.07 |
| Finetune 3 | 20 | 0.0001 | 1.99 |
| **Finetune 4** | **30** | **0.0001** | **1.89** |

Training setup:

```
Optimizer: Adam
Scheduler: ReduceLROnPlateau
Loss: CrossEntropy
```

---

# 🛠 Local Setup

Requirements:

- Python 3.12
- CUDA GPU (optional)
- FluidSynth
- SoundFont (.sf2)

---

## Installation

```bash
git clone https://github.com/Atharva130/MusicLSTM.git
cd MusicLSTM

python -m venv mlenv
mlenv\Scripts\activate

pip install -r requirements.txt
```

---

## Download Dataset

```bash
python src/download_midi.py
```

---

## Preprocess Dataset

```bash
python src/preprocess.py
```

---

## Train Model

```bash
python src/train.py
```

---

## Run Application

```bash
python app.py
```

Open:

```
http://localhost:7860
```

---

# 📂 Project Structure

```
MusicLSTM
│
├── app.py
├── config.yaml
├── requirements.txt
│
├── src
│   ├── models
│   │   ├── lstm_model.py
│   │   ├── rnn_model.py
│   │   ├── gru_model.py
│   │   └── transformer_model.py
│   │
│   ├── preprocess.py
│   ├── download_midi.py
│   ├── train.py
│   ├── train_all.py
│   ├── generate.py
│   └── midi_to_audio.py
│
└── checkpoints
    └── finetune_epoch_30.pt
```

---

# ⚠️ Current Limitations

- Mood conditioning is **rule-based**
- Chords simplified to lowest pitch
- Model can repeat short phrases
- No polyphonic generation
- Vocabulary limited to training distribution

---

# 🔮 Future Improvements

- Conditional generation with **mood embeddings**
- Expand dataset to **2000+ MIDI files**
- Polyphonic tokenization
- Neural audio synthesis
- MIDI export for DAW editing

---

# 🧠 Tech Stack

```
PyTorch
music21
FluidSynth
Gradio
Python
```

---

# 🎼 Inspired By

- Johann Sebastian Bach  
- Ludwig van Beethoven  
- Frédéric Chopin  

---

⭐ If you like this project, consider **starring the repository**.
