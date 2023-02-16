# build tree sitter library
# By Lei Chai, 2021-12-12
# the grammar lib located in `~/data/grammar`

from tree_sitter import Language


grammar_root = '/home/chail/data/grammar'
lang_list = ['python', 'java', 'go']
Language.build_library(
    'mmlang.so',
    [f'{grammar_root}/tree-sitter-{lang}' for lang in lang_list]
)
