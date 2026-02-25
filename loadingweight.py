#yo code chai load garda error handle garna ko lagi...

import os
import re
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers import TextVectorization

# Import our model loading module
from Loads_model import load_trained_model, IMAGE_SIZE, SEQ_LENGTH, VOCAB_SIZE

# Path to the saved weights
WEIGHTS_PATH = "Saved_model/50epoch.weights.h5"
# Load the model
try:
    print("Loading model...")
    model = load_trained_model(WEIGHTS_PATH)
    print("Model loaded successfully!")
    
    # Create dummy data for testing
    test_img = tf.random.normal((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    test_seq = tf.random.uniform((1, SEQ_LENGTH), maxval=VOCAB_SIZE, dtype=tf.int32)
    
    # Test model prediction
    print("Testing model prediction...")
    output = model((test_img, test_seq))
    print(f"Model output shape: {output.shape}")
    print("Model is working correctly!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    
# model.summary()