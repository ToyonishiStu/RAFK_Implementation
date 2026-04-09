# ベースイメージ（CUDA 12.8 + Ubuntu 22.04）
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# 非対話モード（インストール時の質問をスキップ）
ENV DEBIAN_FRONTEND=noninteractive

# 基本ツールのインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    git \
    cuda-nvcc-12-8 \
    libcusparse-dev-12-8 \
    && rm -rf /var/lib/apt/lists/*

RUN for pkg in cublas cufft curand cusolver cusparse cuda_runtime; do \
        find /usr/local/lib/python3.10/dist-packages/nvidia/$pkg/include -maxdepth 1 -name "*.h" 2>/dev/null | \
        while read h; do fname=$(basename "$h"); [ ! -e "/usr/local/cuda/include/$fname" ] && ln -s "$h" "/usr/local/cuda/include/$fname"; done; \
    done

# Python のパス統一
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

# 依存パッケージのインストール（必要なら requirements.txt を追加）
RUN pip install --no-cache-dir torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt /tmp/requirements.txt
# uv のリゾルバのほうが深い依存グラフ（TF / Waymo など）でも収束しやすい
RUN pip install --upgrade pip && \
    pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -r /tmp/requirements.txt

# ワークディレクトリ
WORKDIR /workspaces

# デフォルトコマンド
CMD ["/bin/bash"]
