# PA-former
Official implementation of our paper, Pyramid Attention For Source Code Summarization.


## Getting Started

### Dataset

1. download `data.zip` from https://www.dropbox.com/s/k84cz704ld44yqi/data.zip?dl=0
2. unzip `data.zip`, and we have `./data/*.jsonl`.

### Installs

1. install `pytorch`(>=1.10.x is recommended) according to your environment
2. `pip install -r doc/requirements.txt`

### Training

```bash
python main.py --cfg configs/pa_former/pa_java.yaml --cuda <gpu ids>
```

The training log will be stored in `./out/pa_former/`.
