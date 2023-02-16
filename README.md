# PA-former
Official implementation of our paper, Pyramid Attention For Source Code Summarization.


## Getting Started

### Dataset

1. zipped [EMSE-Deepcom](https://drive.google.com/file/d/1pKf_ji5OstEPzC_RcgTQq4yyvjnvmxqN/view?usp=share_link) and [FunCom](https://drive.google.com/file/d/1-WSPQNAnoW4QFd0Hi4tc6ellrVTZx_Bo/view?usp=share_link) on Google Drive.
2. unzip `*.zip`, and we have `./data/*_train.jsonl` and `./data/*_test.jsonl`.

### Installs

1. install `pytorch`(>=1.10.x is recommended) according to your environment
2. `pip install -r doc/requirements.txt`

### Training

```bash
python main.py --cfg configs/pa_former/pa_java.yaml --cuda <gpu ids>
```

The training log will be stored in `./out/pa_former/`.

### Developing

Go [Getting Started](/doc/getting_started.md)