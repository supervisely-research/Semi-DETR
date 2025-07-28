```bash
pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install yapf==0.32.0 tensorboard future

apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libjpeg-turbo8 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*
```