# Start with RunPod's PyTorch image
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set workspace directory
WORKDIR /workspace

# Create virtual environment
RUN python -m venv /workspace/venvs/.tiny_zero

# Set environment to use venv by default
ENV PATH="/workspace/venvs/.tiny_zero/bin:$PATH"
ENV VIRTUAL_ENV="/workspace/venvs/.tiny_zero"

# Clone TinyZero repository
RUN git clone https://github.com/RobertMcCarthy97/TinyZero.git
WORKDIR /workspace/TinyZero

# Install dependencies in exact order in the venv
RUN . /workspace/venvs/.tiny_zero/bin/activate && \
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install vllm==0.6.3 && \
    pip install ray && \
    pip install -e . && \
    pip install wheel && \
    pip install flash-attn --no-build-isolation && \
    pip install wandb IPython matplotlib

# Run data preprocessing scripts
RUN . /workspace/venvs/.tiny_zero/bin/activate && \
    python ./examples/data_preprocess/countdown.py --local_dir data/countdown && \
    python ./examples/data_preprocess/arth.py --local_dir data/arth_default && \
    python ./examples/data_preprocess/arth_simple.py --local_dir data/arth_simple && \
    python ./examples/data_preprocess/arth_super_simple.py --local_dir data/arth_super_simple

# Create entrypoint script to activate venv
RUN echo '#!/bin/bash\nsource /workspace/venvs/.tiny_zero/bin/activate\nexec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]