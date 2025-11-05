import tensorflow as tf
import sys

print(f"Python Version: {sys.version}")
print(f"TensorFlow Version: {tf.__version__}")
print("-" * 30)

# This is the core command to check for the GPU
gpu_devices = tf.config.list_physical_devices('GPU')

if len(gpu_devices) > 0:
    print("✅ TensorFlow **is** recognizing the GPU.")
    print("Available GPU(s):")
    for device in gpu_devices:
        print(f"  - {device}")
else:
    print("❌ TensorFlow **is not** recognizing the GPU.")
    print("   Make sure you have installed the 'tensorflow-metal' package.")

print("-" * 30)