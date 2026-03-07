# 🎹 MusicAI — Mood-Based Classical Music Generator

> A 2-layer stacked LSTM with Bahdanau Attention trained on 501 classical MIDI files from Bach, Beethoven & Chopin. Select a mood → generate an original piano composition → download as audio.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square)
![Gradio](https://img.shields.io/badge/Gradio-6.x-orange?style=flat-square)

---

## 🎯 What It Does

Users select one of 5 moods. The model generates a novel piano composition in that style, rendered as a playable audio file — entirely from scratch, no GenAI APIs.

| Mood | Key | BPM | Influence |
|------|-----|-----|-----------|
| 🌿 Calm | Major | 76 | Bach |
| 🌧️ Sad | Minor | 60 | Chopin |
| 🌹 Romantic | Major | 72 | Beethoven |
| ⚡ Energetic | Major | 140 | Beethoven |
| 🌑 Dark | Minor | 88 | Bach |

---

## 🏗️ Architecture

```
Input Sequence (seq_len=64)
        ↓
Embedding(577, 64)
        ↓
LSTM Layer 1  (hidden=512, dropout=0.3)
        ↓
LSTM Layer 2  (hidden=512)
        ↓
Bahdanau Attention
        ↓
Linear(512 → 577)
        ↓
Softmax + Temperature Sampling
        ↓
Generated Token Sequence
```

**Total parameters: 3,875,441**

Token format: `{pitch}_{duration}` — e.g. `C4_0.5`, `G#3_1.0`
The model learns pitch AND rhythm simultaneously from a unified vocabulary of 577 tokens.

---

## 📊 Ablation Study

All models trained on the same dataset, same hyperparameters, same sequence length (64).

| Model | Best Loss | Best Accuracy | Notes |
|-------|-----------|---------------|-------|
| Vanilla RNN | 5.43 | 1.31% | ❌ Vanishing gradient on seq_len=64 |
| GRU | 3.79 | 15.82% | ⚠️ Peaks at epoch 3, overfits after |
| Transformer | 3.89 | 13.60% | ⚠️ Insufficient data for attention heads |
| **LSTM + Attention** | **1.89** | **~37%** | ✅ Clear winner |

**Why LSTM beat Transformer:** With only 501 MIDI files (~375K sequences), the Transformer's self-attention lacks sufficient data to generalize. The LSTM's inductive bias toward sequential patterns is a better fit at this data scale.

**Why RNN failed:** Vanilla RNN suffers from vanishing gradients on sequences of length 64. Loss plateaued at 5.4 — essentially random prediction. LSTM's gating mechanism explicitly controls gradient flow.

---

## 🗂️ Dataset

- **Bach:** 410 files (via music21 corpus)
- **Beethoven:** 30 files (music21 + manual collection)
- **Chopin:** 61 files (public domain MIDI)
- **Total: 501 MIDI files → 375,552 tokens → 577 unique tokens**

Preprocessing pipeline:
1. Parse MIDI/MXL with `music21`
2. Extract note + duration pairs → `pitch_duration` tokens
3. Round durations to nearest 0.25, clamp to [0.25, 2.0]
4. Simplify chords to lowest note
5. Build vocabulary → sliding window sequences (length 64, stride 1)

---

## 🎼 Generation Pipeline

```
Seed Sequence (64 tokens from training data)
        ↓
Autoregressive LSTM inference
        ↓
Temperature sampling (controls creativity)
        ↓
Mood-based scale filtering (major/minor)
        ↓
Chord injection every 4th note
        ↓
Dynamic velocity curve (soft → loud → soft)
        ↓
±5% duration humanization
        ↓
music21 → MIDI → FluidSynth → WAV
```

---

## 🚀 Training

Trained in multiple fine-tuning rounds on an NVIDIA RTX 4050:

| Stage | Epochs | Learning Rate | Final Loss |
|-------|--------|--------------|------------|
| Initial | 50 | 0.001 | 2.52 |
| Finetune 1 | 20 | 0.0001 | 2.18 |
| Finetune 2 | 20 | 0.0001 | 2.07 |
| Finetune 3 | 20 | 0.0001 | 1.99 |
| **Finetune 4** | **30** | **0.0001** | **1.89 ✅** |

Optimizer: Adam | Scheduler: ReduceLROnPlateau (factor=0.5, patience=3) | Loss: CrossEntropy

---

## 🛠️ Local Setup

### Prerequisites
- Python 3.12
- CUDA-compatible GPU (optional but recommended)
- [FluidSynth](https://www.fluidsynth.org/) + a `.sf2` soundfont

### Install

```bash
git clone https://github.com/Atharva130/MusicLSTM.git
cd MusicLSTM
python -m venv mlenv
mlenv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Download Data & Preprocess

```bash
python src/download_midi.py
python src/preprocess.py
```

### Train

```bash
python src/train.py
```

### Run the App

```bash
python app.py
# Open http://localhost:7860
```

---

## 📁 Project Structure

```
MusicLSTM/
├── app.py                        # Gradio web UI
├── config.yaml                   # All hyperparameters
├── requirements.txt
├── src/
│   ├── models/
│   │   ├── lstm_model.py         # MusicLSTM + BahdanauAttention
│   │   ├── rnn_model.py          # Baseline RNN
│   │   ├── gru_model.py          # Baseline GRU
│   │   └── transformer_model.py  # Baseline Transformer
│   ├── preprocess.py             # MIDI → token sequences
│   ├── download_midi.py          # Dataset acquisition
│   ├── train.py                  # Training loop
│   ├── train_all.py              # Ablation study trainer
│   ├── generate.py               # Autoregressive generation
│   └── midi_to_audio.py          # MIDI → WAV via FluidSynth
└── checkpoints/
    └── finetune_epoch_30.pt      # Best checkpoint (loss 1.89)
```

---

## 💡 Key Design Decisions

**Token representation:** Combined `pitch_duration` tokens instead of separate vocabularies. This lets the model learn rhythmic patterns tied to specific pitches — e.g. Bach's running eighth notes on C major differ from Chopin's sustained minor chords.

**Bahdanau Attention:** Added on top of the LSTM to allow the model to attend back to earlier parts of the sequence when predicting the next note — mimicking how composers reference earlier motifs.

**Mood conditioning:** Implemented via music theory constraints (scale filtering + BPM) rather than learned labels. With 501 files, training a conditional model would require mood-labeled data we don't have. Rule-based conditioning guarantees musical correctness.

**Fine-tuning strategy:** Staged fine-tuning with decaying LR consistently outperformed single long runs and avoided instability.

---

## ⚠️ Flaws 
- Mood conditioning is rule-based (scale filter + BPM), not learned — the model itself has no awareness of mood, only the post-processing does
- Chords are simplified to the lowest note during preprocessing, losing harmonic richness
- Generated music can sometimes repeat short phrases in loops due to the greedy nature of autoregressive sampling
- No polyphony — only single-note melody lines, no true multi-voice piano texture
- Vocabulary of 577 tokens covers only the training distribution — unusual note combinations outside Bach/Beethoven/Chopin style will never be generated

---

## What Can Be Done Next

- Train a conditional LSTM/Transformer with mood as a learned embedding — true end-to-end mood conditioning
- Increase dataset to 2000+ MIDI files across more composers for better generalization
- Add polyphony support by tokenizing chords as full voicings instead of reducing to lowest note
- Replace FluidSynth with a neural audio vocoder (e.g. DiffWave) for more realistic piano sound
- Add a MIDI download button alongside the WAV output
