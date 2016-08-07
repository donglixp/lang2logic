# Setup and Usage

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

- Run experiments
```sh
./run.sh [seq2seq|seq2tree] [jobqueries|geoqueries|atis] [lstm|attention] GPU_ID
```
(At least one GPU card is required.)

```sh
# run seq2seq without attention
./run.sh seq2seq jobqueries lstm
# run seq2seq with attention
./run.sh seq2seq jobqueries attention
```

- Print results
```sh
cat seq2seq/jobqueries/dump_lstm/model.t7.sample
cat seq2seq/jobqueries/dump_attention/model.t7.sample
```
