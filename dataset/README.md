# Data Process

This dir contains all codes about how to raw textual `code` into 
`structure_rep.`, like:

<img src="/assets/data_structure_parser.png" width="700"  alt="structure parser"/>

For more details about multi-granular code rep., please refer to 
[Multi-granular Hierarchical Code Representation](/docs/multi_gran_hier_code_rep.md)

## File Structure of Dataset

Suppose that the dataset is stored in `data_root`, we have:
```
<data_root>
├── java/
│   ├── train.jsonl  # 415389 + 6(error)
│   ├── test.jsonl   # 13237
│   └── valid.jsonl
└── python/
    ├── train.jsonl  # 
    ├── test.jsonl
    └── valid.jsonl
```

## Usage

### Pipeline for data process

- structure parser
  - ast parser based on `tree_sitter`, code in [ast_parser](/dataset/data_process/structure_parser/ast_parser.py)
  - cfg parser implemented by ourselves, code in [cfg_parser](/dataset/data_process/structure_parser/cfg_parser.py)
  - ddg parser (planning)
- vectorize data
  - sub_token sequence
  - token sequence
  - statement sequence
- filter data
- pack the data
- build vocabulary

### Structure parser

We provide visualization method for structure analysis and debug, and we highly
recommend using it in `Jupyter Notebook`:

```python
# `cd` into `<mmcs-root>/dataset` 
from structure_parser import recover_and_check_structure, MultiLangStructureParser

parser = MultiLangStructureParser('java')  # get the structure parser
code = """text of java code"""  # code to be parsed

re = parser.parse_code(code)
ast, cfg = re['ast'], re['cfg']  # ast: treelib.Tree, cfg: networkx.DiGraph

# stm_subtrees: List(treelib.Tree), cfg_edges: List(edge)
stm_subtrees, cfg_edges = parser.split_statements(ast, cfg)
# linear_data: dict, keys: sub_token_seq, token_seq, stm_seq, sub_token_to_token, token_to_stm
linear_data = parser.linear_structure_data(ast, stm_subtrees)
# gz: graphviz.Digraph
gz = recover_and_check_structure(linear_data, cfg_edges)
```

We can visualize the `gz` to check or analysis.

### Vectorize data

We provide the script to convert the `textual` code into `vectorized` multi-scale structure
representation. 

```bash
python vectorize_data.py --src_root <root of src> --tgt_root <root of tgt> --split <train/test> --dataset <java/python>
```

This script will convert the raw data `<src_root>/<dataset>/<split>.jsonl` into 
`<tgt_root>/<dataset>/linear_<split>.jsonl`. 

### Filter dataset

We select reasonable examples from the full dataset (`linear_*.jsonl`) and save them into `final_*.jsonl`.

```bash
python filter_data.py --src_root <root of src> --dataset <java/python>
```

