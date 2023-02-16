""" utils for data process
by Lei Chai, 2021-12-12
"""

import re
import treelib
import graphviz


def _check_all_upper(token):
    char_list = list(filter(lambda x: x.isalpha(), token))
    return all(char.isupper() for char in char_list)


def _get_lower_case_token(token):
    if _check_all_upper(token):
        return token

    new_token = []
    upper_char_cache = []
    if token[0].isupper():
        upper_char_cache.append(token[0])
    else:
        new_token.append(token[0])
    for tok in token[1:]:
        if tok.isupper() and len(upper_char_cache) > 0:
            upper_char_cache.append(tok)
        elif tok.islower() and len(upper_char_cache) > 0:
            new_token.append('_')
            new_token.extend(upper_char_cache[:-1])
            new_token.append('_')
            new_token.append(upper_char_cache[-1])
            new_token.append(tok)
            upper_char_cache.clear()
        elif len(upper_char_cache) == 0 and tok.isupper():
            upper_char_cache.append(tok)
        else:
            new_token.append(tok)
    if len(upper_char_cache) > 0:
        new_token.append('_')
        new_token.extend(upper_char_cache)
    return ''.join(new_token)


def get_subtoken(token):
    if len(token) == 0:  # return the empty token
        return token
    if token[0] == token[-1] == "'":
        return [token]  # string
    token = _get_lower_case_token(token)
    subtoken = [tok.lower() for tok in re.split('[ _\t]', token)]
    
    return list(filter(lambda x: len(x) > 0, subtoken))


def divide_text(text, en_dict):
    final_words = []
    
    for i in range(len(text)-3):
        left_word, right_word = text[:i+2], text[i+2:]
        words = []
        
        if en_dict.check(left_word):
            words.append(left_word)
        else:
            words += divide_text(left_word, en_dict)
            
        if en_dict.check(right_word):
            words.append(right_word)
        else:
            words += divide_text(right_word, en_dict)
            
        if ''.join(words) == text and (len(words) <= len(final_words) or len(final_words) == 0):
            final_words = words
            
    return final_words


def visualize_cfg(stm_texts, stm_tags, cfg_edges, vi_cfg_path=None):
    """visualize cfg for checking and analysis"""
    from graphviz import Digraph
    gz = Digraph(name='cfg', format='png')

    # add nodes
    for idx, text_tokens in enumerate(stm_texts):
        label = ''
        for pos in range(len(text_tokens) // 40):
            label += text_tokens[40*pos: 40*(pos+1)] + '\n'
        label += text_tokens[40*(len(text_tokens) // 40):]
        node_label = f'{stm_tags[idx]}\n--------\n{label}'
        gz.node(name=f'{idx}', label=node_label, shape='box')
    gz.node(name=f'{len(stm_tags)}', label='end_node')

    # add edges
    for (s, e) in cfg_edges:
        gz.edge(f'{s}', f'{e}')

    if vi_cfg_path is not None:
        gz.render(vi_cfg_path)

    return gz


def recover_and_check_structure(linear_data: dict, cfg_edges: list) -> graphviz.Digraph:
    sub_token_seq = linear_data['sub_token_seq']
    token_seq = linear_data['token_seq']
    stm_seq = linear_data['stm_seq']
    sub_token_to_token = linear_data['sub_token_to_token']
    token_to_stm = linear_data['token_to_stm']

    # print token
    print(sub_token_seq)
    print(sub_token_to_token)
    print(token_seq)
    print(token_to_stm)
    print(stm_seq)
    print(cfg_edges)

    stm_texts = [''] * len(stm_seq)
    for stm_idx, token in zip(token_to_stm, token_seq):
        stm_texts[stm_idx] += ' ' + token

    return visualize_cfg(stm_texts, stm_seq, cfg_edges)


def recover_tree(leaf_seq, non_terminal_seq, start_idx, end_idx):
    ast = treelib.Tree()
    
    ast.create_node(
        tag=non_terminal_seq[0],
        identifier=len(leaf_seq),
        parent=None
    )
    for i, j in zip(start_idx, end_idx):
        if j >= len(leaf_seq):
            tok = non_terminal_seq[j - len(leaf_seq)]
        else:
            tok = leaf_seq[j]
                
        ast.create_node(tag=tok, identifier=j, parent=i)

    return ast
