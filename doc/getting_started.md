# Getting started

## Requirement

```bash
# install `torch` and `dgl` according to your environment

# install python dependency
pip install -r ./doc/requirement.txt  
```

## Usage

We need 3 steps to use this framework for your `seq-generation` task:

- convert your dataset into `jsonl` files
- register your data process methods (python class) into `PIPELINES`
- register your dataset class into `DATASETS`
- register your model class into `MODELS`

### 1. Preparing data

It is highly recommended storing the data using `jsonl` file, where each
example is a `dict` object as a line in the `jsonl` file. For example:

```python
ex = {
    'code': 'public static boolean ...',
    'summary': '...'
}
```

We provide `parallel` script for efficient data pre-processing.
```bash
cd preprocess
python <your-process>.py --dataset <java/python> --split <train/test>
```

If you already have `jsonl` dataset, go directly into step 3.

### 2. Implementing and registering DataReader class

The data_loading class (e.g. [LoadExamplesFromJsonl](/dataset/pipeline/loading.py)) defines
how we load the data examples from files, which should be defined as:

```python
from dataset.build import PIPELINES


@PIPELINES.register_module()
class MyLoadExamplesFromFile:
    def __int__(self, cfg):
        self.init_cfg = copy.deepcopy(cfg)
        self.data_root = cfg.data_root
        # do something
        pass
    
    def __call__(self, pipeline, logger) -> dict:
        # read the local files according to self.cfg 
        # return two lists of examples: train_exs, dev_exs
        pass
        
```

Then, we need a `cfg` to **tell** the information this class needed. just set
the information in a `yaml` file:

```yaml
data:
  loading:
    name: '<name of the loading class, like MyLoadExamplesFromFile>'
    data_root: `<root dir of data>`
    # more cfgs will used by the specified loading class, here is MyLoadExamplesFromFile
```

### 3. Implementing and registering data_preprocessing class

The data_preprocessing class (e.g. [PreprocessBase](/dataset/pipeline/base_data.py)) defines
how we preprocess the loaded raw examples by `DataReader`, which should be defined as:

```python
from dataset.build import PIPELINES


@PIPELINES.register_module()
class MyPreprocess:
    def __int__(self, cfg):
        self.init_cfg = copy.deepcopy(cfg)
        # do something
        pass
    
    def __call__(self, raw_ex, tgt_ex=None) -> dict:
        # extract information from raw_ex and save it to tgt_ex
        pass
        
```

Then, we need a `cfg` to **tell** the information this class needed. just set
the information in a `yaml` file:

```yaml
data:
  pipeline:
    name: '<name of the Preprocess class, like MyPreprocess>'
    # more cfgs will used by the specified preprocessing class, here is MyPreprocess
```

And we provide `Compose` to compose several preprocess methods sequentially, you can use it as:

```yaml
data:
  pipeline:
    name: Compose
    pipelines:
      MyPreprocess1:
        # cfgs for class MyPreprocess1
      MyPreprocess2:
        # cfgs for class MyPreprocess2
      # more data_preprocessing class
      # more cfgs will used by the specified preprocessing class, here is MyPreprocess
```

### 3. Implementing and registering dataset class

The datasets define how to convert the preprocessed data example into `tensor` and how to pack
a batch of converted `tensor`s. We recommend to implements your `Dataset` via inheriting `X2Seq.BaseDataset`, which should be defined as:

```python
from dataset.base_dataset import BaseDataset
from dataset.build import DATASETS


@DATASETS.register_module()
class MyDataset(BaseDataset):
    def __init__(self, config, examples, vocab1, vocab2, ...):
        super(MyDataset, self).__init__(config, examples)
        # init you dataset

    def __getitem__(self, idx) -> dict:
        # how to convert one example into tensors
        pass

    def lengths(self):
        # return lengths of self.examples, which is used for batching the whole dataset
        pass

    @staticmethod
    def collect_fn(batch: list) -> dict:
        # how to batch the examples
        pass
```

Then, we need a `cfg` to **tell** the information this class needed. just set
the information in a `yaml` file:

```yaml
data:
  name: MyDataset
  # cfgs ...
  vocabs:
    vocab1:
      size: 1000
      fields: ['stok_seq']
      no_special_token: True
    vocab2:
      # ...
    # ...
```

### 3. Implementing and registering model class

```python
from models.build import MODELS
from models.base_model import BaseModel


@MODELS.register_module()
class MyModel(BaseModel):
    def __init__(self, config):
        super(MyModel, self).__init__()
        # init model components

    def forward(self, ...):
        # the forward pass for batch training

    def encode(self, ..., return_dict=False):
        # forward pass of encoder

    def step_wise_decode(self, ...):
        # one step of auto-regression decoding, use for inference
```

```yaml
model:
  name: MyModel
  meta_info: # the out of encoder used for decoding attn computation
  # ...
  forward_params:
    # <param name in forward()>: <param name in batch>
    # ...
  predict_params:
    # <param name in encode()>: <param name in batch>
  beam_params:
    # params for beam search, more details in evaluation
    stok_seq:
      data: stok_seq_rep
      mask: stok_pad_mask
```
