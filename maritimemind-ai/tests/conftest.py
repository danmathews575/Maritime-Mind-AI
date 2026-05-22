import os

# Disable TensorFlow backend in HuggingFace transformers to prevent 
# protobuf incompatibilities and speed up model loading.
# SentenceTransformers relies exclusively on PyTorch.
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
