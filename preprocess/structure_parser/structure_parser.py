""" Multi-language structure parser
By Lei Chai, 2021-12-12
"""

import os
import copy
import string
from typing import Optional
import networkx
import treelib
import numpy as np
import networkx as nx

from .ast_parser import AST
from .cfg_parser import JavaCFG, PythonCFG
from .dfg_parser import DFG
from .utils import get_subtoken


class MultiLangStructureParser:
    def __init__(self, language=None):
        assert language in ['python', 'java'], f'un-support {language}'
        file_dir = os.path.dirname(os.path.realpath(__file__))
        os.chdir(file_dir)
        self.language = language
        self.ast_parser = AST(language)
        self.cfg_parser = self._set_cfg_parser(language)
        self.dfg_parser = DFG(language)
        self.max_distance = 32

    @staticmethod
    def _set_cfg_parser(language):
        if language == 'java':
            return JavaCFG()
        
        if language == 'python':
            return PythonCFG()

        raise RuntimeError

    def parse_code(self, code: Optional[str]) -> dict:
        if code is None:
            code = self.get_demo_code()

        code_bytes = bytes(code, 'utf-8')
        ast = self.ast_parser.gen_ast(code_bytes)
        cfg = self.cfg_parser.gen_cfg(ast)
        dfg = self.dfg_parser.gen_dfg(ast, cfg)

        return {
            'ast': ast,
            'cfg': cfg,
            'dfg': dfg
        }

    def parse_cfg(self, code: Optional[str]) -> dict:
        if code is None:
            code = self.get_demo_code()

        code_bytes = bytes(code, 'utf-8')
        ast = self.ast_parser.gen_ast(code_bytes)
        cfg = self.cfg_parser.gen_cfg(ast)

        return {
            'ast': ast,
            'cfg': cfg
        }

    def parse_ast(self, code):
        code_bytes = bytes(code, 'utf-8')
        ast = self.ast_parser.gen_ast(code_bytes)
        return ast

    @staticmethod
    def to_networkx(ast: treelib.Tree) -> nx.DiGraph:
        g = nx.DiGraph()
        nodes = [(n.identifier, {'tag': n.tag, 'token': n.data['token']}) for n in ast.all_nodes_itr()]
        g.add_nodes_from(nodes)
        
        for n in ast.all_nodes_itr():
            for cn in ast.children(n.identifier):
                g.add_edge(n.identifier, cn.identifier)
        
        return g

    @staticmethod
    def split_statements(ast: treelib.Tree, cfg: networkx.DiGraph):
        """split statement sub_tree from `ast` according to `cfg`."""
        # get the statement node ast identifier from cfg
        stm_ast_idfs = [ast.root]
        map_node_to_idx = dict(start=0)
        for node in cfg:
            node_attr: dict = cfg.nodes[node]
            ast_idf = node_attr.get('ast_idx', None)
            if ast_idf is not None:
                map_node_to_idx[node] = len(stm_ast_idfs)
                stm_ast_idfs.append(ast_idf)

        # split statement tree from ast
        stm_subtrees = []
        for stm_idf in stm_ast_idfs:
            stm_subtree = copy.deepcopy(ast.subtree(stm_idf))

            # remove statements in this statement
            for idf in stm_ast_idfs:
                if idf != stm_idf and stm_subtree.get_node(idf):
                    stm_subtree.remove_subtree(idf)

            stm_subtrees.append(stm_subtree)

        # convert the edges
        cfg_edges = []
        map_node_to_idx['end'] = len(stm_ast_idfs)
        for u, v in cfg.edges:
            u_idx, v_idx = map_node_to_idx.get(u, None), map_node_to_idx.get(v, None)
            if u_idx is not None and v_idx is not None:
                cfg_edges.append((u_idx, v_idx))

        return stm_subtrees, cfg_edges

    @staticmethod
    def linear_ast_leaves(ast: treelib.Tree):
        # print the tokenized code snippet corresponding to the input tree
        idxs = sorted([leaf.identifier for leaf in ast.leaves()])
        leaf_tokens, leaf_types, leaf_grammar_types = [], [], []
        for idx in idxs:
            node = ast.get_node(idx)

            # token types
            leaf_types.append(node.tag)

            # grammar type (replace the `identifier` using its parent's tag)
            if node.tag == 'identifier':
                p_node = ast.parent(node.identifier)
                leaf_grammar_types.append(p_node.tag)
            else:
                leaf_grammar_types.append(node.tag)

            # token text
            if node.tag == 'string':
                leaf_tokens.append('STR')
            else:
                leaf_tokens.append(ast.get_node(idx).data['token'])

        return idxs, leaf_tokens, leaf_types, leaf_grammar_types

    def linear_structure_data(self, ast, stm_subtrees):
        """get sub_token_seq, token_seq, stm_seq"""
        # get token sequence
        ast_idx_seq, token_seq, token_types, token_grammar_types = self.linear_ast_leaves(ast)

        # get sub_token sequence and
        # the dependency between token and sub_token
        sub_token_to_token_idx = []
        sub_token_seq = []
        for idx, token in enumerate(token_seq):
            sub_tokens = get_subtoken(token)
            sub_token_seq += sub_tokens
            sub_token_to_token_idx += [idx] * len(sub_tokens)

        # get statement sequence and the dependency between token and stm
        map_ast_idx_to_stm_idx = {}
        stm_seq = []
        for idx, stm_tree in enumerate(stm_subtrees):
            stm_seq.append(stm_tree.get_node(stm_tree.root).tag)
            ast_idxs, _, _, _ = self.linear_ast_leaves(stm_tree)
            for ast_idx in ast_idxs:
                map_ast_idx_to_stm_idx[ast_idx] = idx
        token_to_stm_idx = [map_ast_idx_to_stm_idx[ast_idx] for ast_idx in ast_idx_seq]

        return {
            'sub_token_seq': sub_token_seq,
            'token_seq': token_seq,
            'token_types': token_types,
            'token_grammar_types': token_grammar_types,
            'stm_seq': stm_seq,
            'sub_token_to_token': sub_token_to_token_idx,
            'token_to_stm': token_to_stm_idx
        }

    @staticmethod
    def get_root_path(ast, cfg):
        def leaf_root_path(ast, idx):
            p_n = ast.parent(idx)
            p = [p_n.identifier]
            while p[-1] != ast.root:
                p_n = ast.parent(p_n.identifier)
                p.append(p_n.identifier)
            return p
        
        def gen_path(root_path):
            return [map_id2idx[ast_id] for ast_id in reversed(root_path)]
        
        stm_ast_idfs = [ast.root]
        for node in cfg:
            node_attr: dict = cfg.nodes[node]
            ast_idf = node_attr.get('ast_idx', None)
            if ast_idf is not None:
                stm_ast_idfs.append(ast_idf)

        leaf_ids = sorted([n.identifier for n in ast.leaves()])
        non_terminal_ids = sorted([n.identifier for n in ast.all_nodes_itr() if not n.is_leaf()])
        num_leaf = len(leaf_ids)
        path_assign = np.zeros((num_leaf, num_leaf), dtype=int)
        root_path = [leaf_root_path(ast, leaf_id) for leaf_id in leaf_ids]
        map_id2idx = {}
        for idx, nt_id in enumerate(non_terminal_ids):
            map_id2idx[nt_id] = idx

        path_list, c, path_assign = [], 0, []
        missed_path = {}
        for i in range(num_leaf):
            idx = missed_path.get(str(root_path[i]), None)
            if idx is None:
                missed_path[str(root_path[i])] = c
                path_assign.append(c)
                path_list.append(gen_path(root_path[i]))
                c += 1
            else:
                path_assign.append(idx)
                
                
        non_terminal_seq = [ast.get_node(idx).tag for idx in non_terminal_ids]
        leaf_seq = [ast.get_node(idx).tag for idx in leaf_ids]
        stm_in_non_terminal = [map_id2idx[idf] for idf in stm_ast_idfs]
        
        return {
            'tok_seq': leaf_seq,
            'nt_seq': non_terminal_seq,
            'stm_in_nt': stm_in_non_terminal,
            'path_assign': path_assign,
            'path_list': path_list
        }

    @staticmethod
    def get_pair_fork_path(ast):
        def leaf_root_path(ast, idx):
            node = ast.get_node(idx)
            if node.tag in string.punctuation:  # punctuation, ignore 
                return None

            p_n = ast.parent(idx)
            p = [p_n.identifier]
            while p[-1] != ast.root:
                p_n = ast.parent(p_n.identifier)
                p.append(p_n.identifier)
            return p

        def merge_path(p1, p2, map_id2idx, max_depth=16):
            p1 = list(reversed(p1))
            p2 = list(reversed(p2))
            inds, values = [], []
            c_idx = min(len(p1), len(p2)) - 1

            for idx in range(min(len(p1), len(p2))):
                if p1[idx] != p2[idx]:
                    c_idx = idx - 1  # the crossing point
                    break

            for p, idx in enumerate(p1[c_idx::-1]):  # cp -> root: 0, -1, ...  
                inds.append(map_id2idx[idx])
                v = max_depth - p
                values.append(v if v >= 0 else 0)
            for p, idx in enumerate(p1[c_idx+1:]):  # cp -> leaf1: 0, 1, 2, ...
                inds.append(map_id2idx[idx])
                v = max_depth + p + 1
                values.append(v if v < max_depth*2 else max_depth*2-1)
            for p, idx in enumerate(p2[c_idx+1:]):  # cp -> leaf2: 0, 1, 2, ...
                v = max_depth + p + 1
                inds.append(map_id2idx[idx])
                values.append(v if v < max_depth*2 else max_depth*2-1)

            return inds, values

        leaf_ids = sorted([n.identifier for n in ast.leaves()])
        non_terminal_ids = sorted([n.identifier for n in ast.all_nodes_itr() if not n.is_leaf()])
        num_leaf = len(leaf_ids)
        path_assign = np.zeros((num_leaf, num_leaf), dtype=int)
        root_path = [leaf_root_path(ast, leaf_id) for leaf_id in leaf_ids]
        map_id2idx = {}
        for idx, nt_id in enumerate(non_terminal_ids):
            map_id2idx[nt_id] = idx

        path_list, c = [([], [])], 1
        missed_path = {}
        for i in range(num_leaf):
            if root_path[i] is None:  # punctuation, ignore
                path_assign[i, :] = 0
                continue
            for j in range(i, num_leaf):
                if root_path[j] is None:  # punctuation, ignore
                    path_assign[i, j] = 0
                    continue
                path = merge_path(root_path[i], root_path[j], map_id2idx)
                s_path = str(path)
                if s_path in missed_path:
                    path_assign[i,j] = missed_path[s_path]
                else:
                    missed_path[s_path] = c
                    path_assign[i,j] = c
                    path_list.append(path)
                    c += 1
                
        non_terminal_seq = [ast.get_node(idx).tag for idx in non_terminal_ids]
        leaf_seq = [ast.get_node(idx).tag for idx in leaf_ids]

        # compress
        c_path_assign = []
        for i, row in enumerate(path_assign.tolist()):
            c_path_assign += row[i:]

        return leaf_seq, non_terminal_seq, c_path_assign, path_list
        
    @staticmethod
    def linear_ast_sbt(ast):
        """ linear the ast using structure-based traversal (SBT), or deep-first traversal,
        implementation of <https://xin-xia.github.io/publication/icpc182.pdf>
        Args:
            ast (treelib.Tree): the parsed AST
        Returns:
            - List[str], the generated SBT sequence
        """

        def reformat(tag):
            # example: `method_declaration` -> `MethodDeclaration`
            return ''.join([tok[0].upper() + tok[1:] for tok in tag.split('_')])

        def sbt(node, seq):
            tag = reformat(node.tag)
            if node.is_leaf():
                if tag == node.data["token"]:
                    tok = tag
                else:
                    tok = f'{tag}_{node.data["token"]}'
                seq.append('(')
                seq.append(tok)
                seq.append(')')
                seq.append(tok)
            else:
                seq.append('(')
                seq.append(tag)
                for c_node in ast.children(node.identifier):
                    sbt(c_node, seq)
                seq.append(')')
                seq.append(tag)

        sbt_seq = []
        sbt(ast.get_node(ast.root), sbt_seq)

        return sbt_seq

    @staticmethod
    def linear_ast(ast: treelib.Tree):
        non_terminal_idxs, leaf_idxs = [], []
        for node in ast.all_nodes_itr():
            if node.is_leaf():
                leaf_idxs.append(node.identifier)
            else:
                non_terminal_idxs.append(node.identifier)
        non_terminal_idxs = sorted(non_terminal_idxs)
        leaf_idxs = sorted(leaf_idxs)

        nt_idx2pos, lf_idx2pos = {}, {}
        for i, idx in enumerate(non_terminal_idxs):
            nt_idx2pos[idx] = i
        for i, idx in enumerate(leaf_idxs):
            lf_idx2pos[idx] = i
        
        # edges
        start_idx, end_idx = [], []
        for node in ast.all_nodes_itr():
            for c_node in ast.children(node.identifier):
                if c_node.is_leaf():
                    end_idx.append(lf_idx2pos[c_node.identifier])
                else:
                    end_idx.append(nt_idx2pos[c_node.identifier] + len(leaf_idxs))
                start_idx.append(nt_idx2pos[node.identifier] + len(leaf_idxs))
        edges = (start_idx, end_idx)
                
        non_terminal_seq = [ast.get_node(idx).tag for idx in non_terminal_idxs]
        leaf_seq = []
        for idx in leaf_idxs:
            node = ast.get_node(idx)
            if node.tag == 'string':
                leaf_seq.append('STR')
            else:
                leaf_seq.append(node.data['token'])

        return leaf_seq, non_terminal_seq, edges

    @staticmethod
    def recover_ast(leaf_seq, non_terminal_seq, edges) -> treelib.Tree:
        ast = treelib.Tree()
        start_idx, end_idx = edges
        
        # root node
        ast.create_node(
            tag=non_terminal_seq[0],
            identifier=len(leaf_seq),  # id of non-terminal node starts from `len(leaf_seq)`
            parent=None
        )

        for i, j in zip(start_idx, end_idx):
            # non-terminal idx: [len(leaf_seq), ...]
            end_tok = non_terminal_seq[j - len(leaf_seq)] if j >= len(leaf_seq) else leaf_seq[j]            
            ast.create_node(tag=end_tok, identifier=j, parent=i)

        return ast

    @staticmethod
    def pack_node_to_dict(tree: treelib.Tree, idx, root_idx=None):
        node: treelib.Node = tree.get_node(idx)
        children = sorted([n.identifier for n in tree.children(node.identifier)])
        parent: treelib.Node = tree.parent(idx)
        ex = {
            'id': node.identifier,
            'tag': node.tag,
            'data': node.data,
            'children': children,
            'parent': None if parent is None else parent.identifier,
            'root_idx': root_idx
        }
        return ex

    @staticmethod
    def pack_dfg_to_seq(dfg: nx.MultiDiGraph, ast: treelib.Tree):
        ast_to_flatten = {}
        tok_seq = []
        for ft, ai in enumerate(dfg.nodes):
            ast_to_flatten[ai] = ft
            tok_seq.append(ast.get_node(ai).data['token'])

        stok_to_tok, stok_seq = [], []
        for idx, tok in enumerate(tok_seq):
            stoks = get_subtoken(tok)
            stok_seq += stoks
            stok_to_tok += [idx] * len(stoks)

        edges, edges_type = [], []
        for u, v, t in dfg.edges:
            edge = dfg.edges[u, v, t]
            edges.append((ast_to_flatten[u], ast_to_flatten[v]))
            edges_type.append(edge['type'])
        
        return {
            'tok_seq': tok_seq,
            'stok_seq': stok_seq,
            'stok2tok': stok_to_tok,
            'edges': edges,
            'edge_type': edges_type 
        }

    def get_demo_code(self):
        if self.language == 'python':
            code = """def add(a=3, b=4):
\t"s sample test case"
\treturn a + b  # test inline comment remove
"""
        elif self.language == 'java':
            code = """public class MLASTParser{
    boolean f(Set<String> set, String value) {
        for (String entry : set) {
            if (entry.equalsIgnoreCase(value)) {
                return true;
            }
        }
        return false;
    }
}"""
        else:
            raise NotImplementedError
        return code
