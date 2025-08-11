# POEX

```
conda create -n poex python=3.11
conda activate poex

git clone --recurse-submodules https://github.com/poex-jailbreak/POEX.git
cd POEX/poex
pip install -e .

cd Fastchat
pip install -e .

# set base_url and api key
cd src
python run_poex_qwen.py
```