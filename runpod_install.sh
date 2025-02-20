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

pip install ipykernel jupyter notebook
/workspace/venvs/.tiny_zero/bin/python3 -m ipykernel install --user --name=tiny_zero --display-name="Tiny Zero Environment"


python ./examples/data_preprocess/countdown.py --local_dir data/countdown
python ./examples/data_preprocess/arth.py --local_dir data/arth_default
python ./examples/data_preprocess/arth_simple.py --local_dir data/arth_simple
python ./examples/data_preprocess/arth_super_simple.py --local_dir data/arth_super_simple

python ./examples/data_preprocess/arth_prompt_decompose_example.py --local_dir data/arth_prompt_decompose_example
python ./examples/data_preprocess/arth_prompt_decompose.py --local_dir data/arth_prompt_decompose
python ./examples/data_preprocess/arth_prompt_decompose_instruct.py --local_dir data/arth_prompt_decompose_instruct
python ./examples/data_preprocess/arth_prompt_replace_lvl1_decompose_example.py --local_dir data/arth_prompt_replace_lvl1_decompose_example
python ./examples/data_preprocess/arth_prompt_replace_lvl1_decompose.py --local_dir data/arth_prompt_replace_lvl1_decompose
python ./examples/data_preprocess/arth_super_simple_prompt_decompose_example.py --local_dir data/arth_super_simple_prompt_decompose_example

python ./examples/data_preprocess/arth_instruct_story.py --local_dir data/arth_instruct_story

python ./examples/data_preprocess/arth_direct_answer_instruct.py --local_dir data/arth_direct_answer_instruct

python ./examples/data_preprocess/arth_prompt_replace_lvl2_decompose_example.py --local_dir data/arth_prompt_replace_lvl2_decompose_example

python ./examples/data_preprocess/arth_prompt_replace_vague_lvl1_decompose.py --local_dir data/arth_prompt_replace_vague_lvl1_decompose

python ./examples/data_preprocess/gsm8k_instruct.py --local_dir data/gsm8k_instruct
python ./examples/data_preprocess/gsm8k_instruct_moods.py --local_dir data/gsm8k_instruct_moods

### SYCOPHANCY
# Standard
python ./examples/data_preprocess/sycophancy/sycophancy_core.py --local_dir data/sycophancy
# 2 president standard
python ./examples/data_preprocess/sycophancy/sycophancy_core.py --local_dir data/sycophancy_2_president_standard --num_presidents 2
# 2 president encoded
python ./examples/data_preprocess/sycophancy/sycophancy_core.py --local_dir data/sycophancy_2_president_encoded --num_presidents 2 --prompt_type encoded
# 2 president blatant heavy example
python ./examples/data_preprocess/sycophancy/sycophancy_core.py --local_dir data/sycophancy_2_president_blatant_heavy_example --num_presidents 2 --prompt_type blatant_heavy_example


