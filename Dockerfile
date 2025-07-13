FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /SynapseX

# 1. 安裝系統依賴、Python3、pip、編譯工具，並清理快取
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential libssl-dev libffi-dev libjpeg-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. 升級 pip、wheel、setuptools
RUN pip3 install --upgrade pip wheel setuptools

# 3. 安裝 Python 相依套件
COPY requirements.txt /SynapseX/
RUN pip3 install -r requirements.txt


# 4. 複製並安裝 mamba_ssm wheel    
COPY mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl /SynapseX/
RUN pip3 install mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 5. 安裝 PyTorch (CUDA 12.4)
RUN pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124


RUN pip3 install causal-conv1d>=1.4.0 --no-build-isolation

# 5. 複製專案程式碼
COPY . /SynapseX
COPY ssd_combined.py /usr/local/lib/python3.10/dist-packages/mamba_ssm/ops/triton/ssd_combined.py




# Mamba2

# FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
# ENV DEBIAN_FRONTEND=noninteractive

# WORKDIR /SynapseX

# # 1. 安裝系統依賴、Python3、pip、編譯工具，並清理快取
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#       python3 python3-pip python3-dev build-essential libssl-dev libffi-dev libjpeg-dev zlib1g-dev && \
#     rm -rf /var/lib/apt/lists/*

# # 2. 升級 pip、wheel、setuptools
# RUN pip3 install --upgrade pip wheel setuptools

# # 3. 安裝 Python 相依套件
# COPY requirements.txt /SynapseX/
# RUN pip3 install -r requirements.txt


# # 4. 複製並安裝 mamba_ssm wheel    
# COPY mamba_ssm-2.2.4+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl /SynapseX/
# RUN pip3 install mamba_ssm-2.2.4+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# # 5. 安裝 PyTorch (cu121)
# RUN pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121


# RUN pip3 install causal-conv1d>=1.4.0 --no-build-isolation

# # 5. 複製專案程式碼
# COPY . /SynapseX
# COPY ssd_combined.py /usr/local/lib/python3.10/dist-packages/mamba_ssm/ops/triton/ssd_combined.py