FROM nvidia/cuda:13.3.0-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /SynapseX

# 1. 安裝系統依賴、Python3、pip、編譯工具，並清理快取
RUN apt-get update 
RUN apt-get install -y --no-install-recommends python3 python3-pip python3-dev build-essential libssl-dev libffi-dev libjpeg-dev zlib1g-dev
    
RUN apt install -y python3.12-venv
RUN rm -rf /var/lib/apt/lists/*

# 建立虛擬環境並將其加入 PATH
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 此時更新的是虛擬環境內的 pip，不會觸發 debian 卸載衝突
RUN pip install --upgrade pip wheel setuptools
RUN pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130


COPY . /SynapseX
RUN pip install .


# 4. 複製並安裝 mamba_ssm wheel    
RUN pip install mamba_ssm-2.3.2.post1+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
RUN pip install causal_conv1d-1.7.0+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/nvshmem/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"