# jch-nerf
Jump Consistent Hash Nerf

# Setup
```bash
conda create --name jch_nerf python=3.10
export PATH=/usr/local/cuda-12.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
cd nerfacc
pip install -e .
cd ../tiny-cuda-nn/bindings/torch
pip install .

