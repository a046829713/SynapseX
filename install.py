import sys
import platform
import subprocess
import time


def get_cuda_version():
    """使用 nvcc 指令檢查 CUDA 版本，如果指令不存在則返回 None。"""
    try:
        # 執行 nvcc --version 指令
        output = subprocess.check_output(["nvcc", "--version"], text=True)
        # 在輸出中尋找 'release' 開頭的行
        for line in output.split("\n"):
            if "release" in line and "V" in line:
                # 例如: 'Cuda compilation tools, release 12.2, V12.2.140'
                version_str = line.split(",")[1]  # ' release 12.2'
                version = version_str.strip().split(" ")[1]  # '12.2'
                return version
    except (FileNotFoundError, subprocess.CalledProcessError):
        # 如果 nvcc 不存在或執行失敗
        return None
    return None


def install_dependencies():
    """根據環境動態安裝所有依賴項。"""

    if sys.platform != "linux":
        print("錯誤：此專案目前僅支援 Linux。")
        sys.exit(1)

    py_major = sys.version_info.major
    py_minor = sys.version_info.minor

    cuda_version = get_cuda_version()
    if not cuda_version:
        print("錯誤：找不到 CUDA Toolkit (nvcc)。請確認已正確安裝。")
        sys.exit(1)
    print(f"CUDA 版本: {cuda_version}")

    # --- 根據環境設定安裝變數 ---
    pytorch_index_url = ""
    torch_packages = [
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
    ]
    wheel_urls = []

    if cuda_version.startswith("12."):
        # 假設 CUDA 12.x 都與 PyTorch 的 cu124 wheel 兼容
        pytorch_index_url = "https://download.pytorch.org/whl/cu124"
        
        if py_major == 3 and py_minor == 12:
            print("偵測到 CUDA 12.x 與 Python 3.12，設定對應的 wheel 檔案。")
            wheel_urls = [
                "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
                "https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
            ]
        elif py_major == 3 and py_minor == 10:
            print("偵測到 CUDA 12.x 與 Python 3.10，設定對應的 wheel 檔案。")
            wheel_urls = [
                "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
                "https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
            ]

    elif cuda_version.startswith("11."):
        pytorch_index_url = "https://download.pytorch.org/whl/cu118"
        print("警告：偵測到 CUDA 11.x。此專案可能需要 CUDA 12.x 以獲得完整支援。")
        wheel_urls = []


    # 檢查是否有找到對應的設定
    if not pytorch_index_url or not wheel_urls:
        print(f"錯誤：找不到適用於您環境 (CUDA {cuda_version}, Python {py_major}.{py_minor}) 的預編譯 wheel 檔案。")
        print("請檢查專案需求或考慮從源碼編譯。")
        sys.exit(1)

    # --- 開始執行安裝 ---
    try:
        print("\n--- 步驟 1/3: 安裝 PyTorch, torchvision, torchaudio ---")
        pip_command_torch = [sys.executable, "-m", "pip", "install"] + torch_packages + ["--index-url", pytorch_index_url]
        print(f"執行指令: {' '.join(pip_command_torch)}")
        subprocess.check_call(pip_command_torch)
        print("PyTorch 安裝成功！")

        print("\n--- 步驟 2/3: 安裝 mamba-ssm 和 causal-conv1d ---")
        pip_command_wheels = [sys.executable, "-m", "pip", "install"] + wheel_urls
        print(f"執行指令: {' '.join(pip_command_wheels)}")
        subprocess.check_call(pip_command_wheels)
        print("特定的 wheel 檔案安裝成功！")

        print("\n--- 步驟 3/3: 從 requirements_common.txt 安裝通用套件 ---")
        pip_command_common = [sys.executable, "-m", "pip", "install", "-r", "requirements_common.txt"]
        print(f"執行指令: {' '.join(pip_command_common)}")
        subprocess.check_call(pip_command_common)
        print("通用套件安裝成功！")

    except subprocess.CalledProcessError as e:
        print(f"安裝過程中發生錯誤: {e}")
        print("請檢查錯誤訊息並重試。")
        sys.exit(1)


if __name__ == "__main__":
    install_dependencies()