# Setup

- If you have already installed Torch7, please rename its folder name.
```sh
mv ~/torch ~/torch_bak
```

- Download Torch7
```sh
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
```

- Replace ~/torch/extra/cunn/lib/THCUNN/ClassNLLCriterion.cu with the one in the ./install folder.

The original ClassNLLCriterion.cu throws an error when the input is 0. We modify this file to make it accept 0.

- Install Torch7
```sh
cd ~/torch
./install.sh
```

- Follow the instructions (in http://torch.ch/docs/getting-started.html) to refresh your env variables.
```sh
# On Linux with bash
source ~/.bashrc
```

- Install dependency
```sh
luarocks install class
pip install path.py
```

- Pull data
```sh
python pull_data.py
```

# Usage

- Run pretrained models
```sh
./pretrain.sh [seq2seq|seq2tree] [jobqueries|geoqueries|atis] [lstm|attention] GPU_ID
```

```sh
# run seq2seq without attention
./pretrain.sh seq2seq jobqueries lstm
# print results
cat seq2seq/jobqueries/dump_lstm/pretrain.t7.sample
# run seq2seq with attention
./pretrain.sh seq2seq jobqueries attention
# print results
cat seq2seq/jobqueries/dump_attention/pretrain.t7.sample
```

- Run experiments
```sh
./run.sh [seq2seq|seq2tree] [jobqueries|geoqueries|atis] [lstm|attention] GPU_ID
```

```sh
# run seq2seq without attention
./run.sh seq2seq jobqueries lstm
# print results
cat seq2seq/jobqueries/dump_lstm/model.t7.sample
# run seq2seq with attention
./run.sh seq2seq jobqueries attention
# print results
cat seq2seq/jobqueries/dump_attention/model.t7.sample
```
# Environment

* OS: Scientific Linux 7.1
* GCC: 4.9.1 20140922 (Red Hat 4.9.1-10)
* GPU: 980 or titan x
* CUDA: 7.5
* Torch7: c0e51b98acbb54e6655343a57152b6e711ffdc2b

The code is only tested on the above environment.
