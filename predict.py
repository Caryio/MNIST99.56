# SHI Yunjiao 3036191025
from keras.models import load_model
from keras.datasets import mnist
import numpy as np

(_, _), (x_test, y_test) = mnist.load_data()
x_test = np.expand_dims(x_test, -1).astype("float32") / 255
y_test = y_test.astype("float32")

model = load_model('best_model.keras')

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {test_loss:.4f}')
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

model.summary()

total_params = model.count_params()
print(f'Total number of parameters: {total_params}')
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# === My running result in my computer: Total number of parameters: 693951; Test accuracy: 99.56% ===
# 2023-12-01 01:10:47.099179: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: SSE SSE2 SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# Test loss: 0.0127
# Test accuracy: 99.56%
# Model: "sequential_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d_3 (Conv2D)           (None, 28, 28, 24)        888       
                                                                 
#  batch_normalization_4 (Bat  (None, 28, 28, 24)        96        
#  chNormalization)                                                
                                                                 
#  activation_4 (Activation)   (None, 28, 28, 24)        0         
                                                                 
#  dropout_4 (Dropout)         (None, 28, 28, 24)        0         
                                                                 
#  conv2d_4 (Conv2D)           (None, 14, 14, 48)        28848     
                                                                 
#  batch_normalization_5 (Bat  (None, 14, 14, 48)        192       
#  chNormalization)                                                
                                                                 
#  activation_5 (Activation)   (None, 14, 14, 48)        0         
                                                                 
#  dropout_5 (Dropout)         (None, 14, 14, 48)        0         
                                                                 
#  conv2d_5 (Conv2D)           (None, 7, 7, 64)          49216     
                                                                 
#  batch_normalization_6 (Bat  (None, 7, 7, 64)          256       
#  chNormalization)                                                
                                                                 
#  activation_6 (Activation)   (None, 7, 7, 64)          0         
                                                                 
#  dropout_6 (Dropout)         (None, 7, 7, 64)          0         
                                                                 
#  flatten_1 (Flatten)         (None, 3136)              0         
                                                                 
#  dense_2 (Dense)             (None, 195)               611715    
                                                                 
#  batch_normalization_7 (Bat  (None, 195)               780       
#  chNormalization)                                                
                                                                 
#  activation_7 (Activation)   (None, 195)               0         
                                                                 
#  dropout_7 (Dropout)         (None, 195)               0         
                                                                 
#  dense_3 (Dense)             (None, 10)                1960      
                                                                 
# =================================================================
# Total params: 693951 (2.65 MB)
# Trainable params: 693289 (2.64 MB)
# Non-trainable params: 662 (2.59 KB)
# _________________________________________________________________
# Total number of parameters: 693951
# Test accuracy: 99.56%