```bash
# Install custom mmdet and Semi-DETR
cd thirdparty/mmdetection && python -m pip install -e .
cd ../../ && python -m pip install -e .

# Build CUDA ops for deformable attention
cd detr_od/models/utils/ops
python setup.py build install
python test.py
cd ../../../..

# PyTorch and MMCV installation
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# Install other dependencies
pip install yapf==0.32.0 tensorboard future
pip install scipy==1.5.4 scikit-learn==0.23.2

# libs for opencv
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