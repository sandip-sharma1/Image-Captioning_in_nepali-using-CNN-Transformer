# Image-Captioning-using-CNN-Transformer


This project implements an end-to-end Image Captioning system that generates 
natural language descriptions for images using a CNN-based feature extractor 
and a Transformer architecture for sequence modeling.

The repository contains complete training, evaluation, inference, and UI modules.

Dataset used-->Flickr8k(Translated to nepali and manually verified open source dataset)
------------------------------------------------------------
1. Project Architecture
------------------------------------------------------------

The model follows a vision-to-language pipeline:

Image
  → CNN Feature Extraction
  → Transformer Model
  → Caption Generation


CNN Component
-------------
- Extracts high-level visual features from input images.
- Converts images into feature embeddings suitable for sequence modeling.

Transformer Component
---------------------
Implemented in `Transformer.py`.

- Multi-head self-attention
- Positional encoding
- Encoder–Decoder architecture
- Cross-attention between image features and caption tokens
- Autoregressive caption generation

The model predicts one word at a time until an end token is generated.


------------------------------------------------------------
2. Repository Structure
------------------------------------------------------------

```
.
├── BLEUscore.py              # BLEU evaluation implementation
├── dataset/                  # Images and caption data
│   ├── add_image_data
│   └── translated_nepali_captions.txt
├── improvedUI.py             # Enhanced graphical interface
├── inference.py              # Caption generation script
├── LICENSE
├── loadingweight.py          # Load pretrained weights
├── Loads_model.py            # Model loading utilities
├── preprocessing.py          # Data preprocessing utilities
├── README.md
├── requirements.txt
├── Saved_model/              # Saved trained model weights
│   └── yourmodel
├── training.py               # Model training script
├── Transformer.py            # Transformer model implementation
└── UI.py                     # Graphical interface for caption generation

```



------------------------------------------------------------
3. Data Preprocessing
------------------------------------------------------------

Handled in `preprocessing.py`.

Includes:
- Image resizing and normalization
- Caption tokenization
- Vocabulary creation
- Padding sequences
- Dataset preparation for training and validation

Captions are converted into integer sequences before training.


------------------------------------------------------------
4. Training
------------------------------------------------------------

Run:

    python training.py

Training process:
- Load and preprocess dataset
- Build Transformer model
- Compute cross-entropy loss
- Backpropagation and optimization
- Save trained model to `Saved_model/`

Hyperparameters such as learning rate, batch size, and number of epochs
can be modified inside `training.py`.


------------------------------------------------------------
5. Evaluation
------------------------------------------------------------

BLEU score evaluation is implemented in `BLEUscore.py`.

Run:

    python BLEUscore.py

The script computes:
- BLEU-1
- BLEU-2
- BLEU-3
- BLEU-4

This measures the quality of generated captions against reference captions.


------------------------------------------------------------
6. Inference
------------------------------------------------------------

To generate a caption for a new image:

    python inference.py --image_path path/to/image.jpg

Process:
1. Load trained model (via `Loads_model.py`)
2. Preprocess input image
3. Pass image through CNN + Transformer
4. Generate caption token-by-token
5. Output final caption


------------------------------------------------------------
7. Graphical User Interface
------------------------------------------------------------

A simple UI is provided in `UI.py`.

Run:

    python UI.py

Features:
- Upload/select image
- Generate caption
- Display result


------------------------------------------------------------
8. Model Weights
------------------------------------------------------------

Trained weights are stored in:

    Saved_model/

To load weights separately:

    python loadingweight.py


------------------------------------------------------------
9. Key Features
------------------------------------------------------------

- CNN-based visual feature extraction
- Full Transformer (Encoder–Decoder)
- Attention-based caption generation
- BLEU score evaluation
- Modular and structured implementation
- Optional GUI for inference

you will find comments on code in nepali because i think no other than neplease are going to see this repo...

------------------------------------------------------------
Author
------------------------------------------------------------

Sandip Sharma
