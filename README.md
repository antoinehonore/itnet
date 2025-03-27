# Fusion of Irregular Multimodal Sensor Data
Code for the paper "ITNet: Irregular Timeseries Data Fusion with Attention Mechanisms" submitted to FUSION 2025

## Pre-requisite
Create a python virtual environment in the way that you want, e.g. :
```
virtualenv -p python3 pyenv
source pyenv/bin/activate
pip install -r requirements.txt
```

## Sensor fusion experiments
1. Write the location of your python interpreter in `cfg.mk`
```bash
PYTHON=pyenv/bin/python
```

2. Compute configuration combinations using the parameter grid in `configs/default.json` (See https://github.com/barketplace/makegrid) 
```bash
make default-init
```
This creates a folder: `results/default/`, and individual config files in `configs/default/`.

3. Run the experiment using the configuration `configs/default/1.json`:
```bash
python main.py -i configs/default/1.json -o results/default/1.pkl --plot --save --show
```

4. Run all configurations sequentially:
```bash
make default 
```



## (deprecated) Causal Linear Cross-attention with non-triangular mask

### What I have and want to do
The goal is to modify the cpp and cuda kernels for causal dot product proposed in https://github.com/idiap/fast-transformers, to be able to use them with non-triangular causal masks.

- The source code for the library, as well as the python binding code is available in the folder `causal_product`. 

- You can run the pre-compiled dot product function from a local copy of the shared cpp library 
```bash
python main.py
```

- The two lines related to (1) compiling `causal_product/causal_product_cpu.cpp` and (2) creating the shared `.so` library are in `causal_product/compile.sh`

#### Problems
- Compiling the cpp library does not work.

```bash
cd causal_product
./compile.sh
```

- I don't know where to start with the cuda kernel (source code in `causal_product/causal_product_cuda.cu`)
