import os

# Disable GPU / oneDNN (safe on Render CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import uvicorn

print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

if __name__ == "__main__":
    uvicorn.run(
        "app.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
