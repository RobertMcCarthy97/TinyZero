mkdir /workspace/venvs
python -m venv /workspace/venvs/.tiny_zero
source /workspace/venvs/.tiny_zero/bin/activate

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
pip install -e .

# flash attention 2
pip install wheel
pip3 install flash-attn --no-build-isolation
# quality of life
pip install wandb IPython matplotlib

python ./examples/data_preprocess/countdown.py --local_dir data/countdown
python ./examples/data_preprocess/arth.py --local_dir data/arth_default
python ./examples/data_preprocess/arth_simple.py --local_dir data/arth_simple
python ./examples/data_preprocess/arth_super_simple.py --local_dir data/arth_super_simple

python ./examples/data_preprocess/arth_prompt_decompose_example.py --local_dir data/arth_prompt_decompose_example
python ./examples/data_preprocess/arth_super_simple_prompt_decompose_example.py --local_dir data/arth_super_simple_prompt_decompose_example