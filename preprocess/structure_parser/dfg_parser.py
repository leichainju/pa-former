""" add more edges into AST mainly according to the data dependency. 
Allamanis et al. Learning to represent programs with graphs. In ICLR'2018.
Hellendoorn et al. Global Relational Models of Source Code. In ICLR'2020.
---------------------------------------------------------
Nodes:
- non-terminal nodes: init as this node's grammar type.
- leaf nodes: init as text token
Edges:
- Child: original edges in AST.
- NextToken: the textual order, e.g. `int`->`x`->`=`->`3`->`;`
- LastRead: `while (x_1) { x_2 = x_3 + 1; }` => `x_1`->`x_2`, `x_2`->`x_3`, `x_3`->`x_1`
- LestWrite: `while (x_1) { x_2 = x_3 + 1; }` => `x_1`->`x_2`, `x_2`->`x_2`, `x_3`->`x_2`
- ComputeFrom: `x = expr(a, b)` => `x`->`a`, `x`->`b`
- LastLexicalUse: chain all uses of the same variable.
- LeafCFG: move down the edge in CFG into begin token of the corresponding node in CFG. 
"""

import copy
import treelib
import networkx as nx
import matplotlib.pyplot as plt


class DFG:
    """ 
    Usage:
    >>> dfg_parser = DFG('java')
    >>> dfg_parser.gen_dfg(ast, cfg)
    """
    def __init__(self, lang='java'):
        # structure rep.
        self.ast: treelib.Tree = None
        self.cfg: nx.DiGraph = None
        self.dfg: nx.MultiDiGraph = None
        self.simple_cfg: nx.DiGraph = None

        # stm-level rep.
        self.stm_subtrees: dict = None  # `stm_idx` -> treelib.Tree
        self.stm_ast: treelib.Tree = None  # stm-level ast tree
        self.vars_per_stm: dict = None  # `stm_idx` -> list[treelib.Node]
        self.connect_stm_paris: dict = None  # (`src_stm_idx`, `tgt_stm_idx`) -> path(list[int])
        
        # variables information
        self.registered_vars: dict = None  # `token` -> list[treelib.Node]
        self.assigned_vars = []  # `token` -> list[treelib.Node] 

        # others
        if lang == 'java':
            self.var_parent = ['formal_parameter', 'variable_declarator']
        elif lang == 'python':
            self.var_parent = ['formal_parameter', 'variable_declarator']
        else:
            raise RuntimeError('Unsupported Language !')
        
    def gen_dfg(self, ast, cfg):
        # reset src
        self.ast = copy.deepcopy(ast)
        self.cfg = copy.deepcopy(cfg)

        # dfg is a multi-digraph, which contains multi-type edges    
        self.dfg = nx.MultiDiGraph()

        # extract structure information
        self.simple_cfg = self.simplify_cfg()
        self.stm_subtrees, self.stm_ast = self.decompose_ast()
        self.connect_stm_paris = self.get_connected_pairs()
        self.registered_vars = {}
        self.vars_per_stm = {}
        self.assigned_vars = []
        self.collect_variables()

        # add edges according to data flow
        self.add_next_token_edge()
        self.add_leaf_cfg_edge()
        self.add_lexical_use()
        self.add_compute_from()
        self.add_last_read()
        self.add_last_write()

        return self.dfg
        
    def decompose_ast(self):
        """ split ast into statement-level sub-trees"""
        # collect stm-nodes' ast idx 
        stm_ast_idxs = [self.ast.root]
        for node in self.cfg.nodes:
            node_attr: dict = self.cfg.nodes[node]
            ast_idx = node_attr.get('ast_idx', None)
            if ast_idx is not None:
                stm_ast_idxs.append(ast_idx)
        
        # split sub_tree
        stm_subtrees, stm_ast = {}, treelib.Tree()
        for stm_root_idx in stm_ast_idxs:
            stm_subtree = copy.deepcopy(self.ast.subtree(stm_root_idx))
            stm_root = self.ast.get_node(stm_root_idx)
            if stm_ast.size() == 0: # the root stm node
                stm_ast.create_node(
                    tag=stm_root.tag,
                    identifier=stm_root_idx,
                    parent=None
                )

            # remove statements in this statement and update stm_ast
            for idf in stm_ast_idxs:
                if idf != stm_root_idx and stm_subtree.contains(idf):
                    stm_subtree.remove_subtree(idf)
                    if stm_ast.contains(idf):  # add `stm_idf -> idf` to stm_ast
                        stm_ast.remove_node(idf)
                    stm_node = self.ast.get_node(idf)
                    stm_ast.create_node(
                        tag=stm_node.tag,
                        identifier=idf,
                        parent=stm_root_idx
                    )

            for n in stm_subtree.all_nodes_itr():
                n.data.update({
                    'stm_tag': stm_root.tag, 
                    'stm_idx': stm_root.identifier
                })

            stm_subtrees[stm_root_idx] = stm_subtree
            
        return stm_subtrees, stm_ast
    
    def get_connected_pairs(self):
        all_raw_paths = nx.all_pairs_shortest_path(self.simple_cfg)
        all_paths = {}
        for src, tgts in filter(lambda p: p[0] not in ['start', 'end'], all_raw_paths):
            tgt_paths = {}
            for tgt, path in tgts.items():
                tgt_paths[tgt] = path
            all_paths[src] = tgt_paths
        return all_paths
        
    def collect_variables(self):
        """ update `self.vars_per_stm` and `self.registered_vars` """
        for stm_root_idx, stm_ast in self.stm_subtrees.items():
            stm_vars = []
            for n in stm_ast.all_nodes_itr():
                if n.tag == 'identifier':
                    p = stm_ast.parent(n.identifier)
                    if n.data['token'] in self.registered_vars:
                        self.registered_vars[n.data['token']].append(n)
                        stm_vars.append(n)
                    elif p.tag in self.var_parent:  # new data 
                        self.registered_vars[n.data['token']] = [n]
                        self.assigned_vars.append(n.identifier)
                        stm_vars.append(n)
            self.vars_per_stm[stm_root_idx] = stm_vars
        
    def add_last_read(self):
        def check(src_stm_idx, tgt_stm_idx):
            return src_stm_idx in self.connect_stm_paris and \
                tgt_stm_idx in self.connect_stm_paris[src_stm_idx]

        for var_chain in self.registered_vars.values():
            for node_src in var_chain:
                last_read = {}
                src_stm_idx, src_idx = node_src.data['stm_idx'], node_src.identifier
                for node_tgt in var_chain:
                    tgt_stm_idx, tgt_idx = node_tgt.data['stm_idx'], node_tgt.identifier
                    
                    if src_idx != tgt_idx and check(src_stm_idx, tgt_stm_idx):
                        curr_path = set(self.connect_stm_paris[src_stm_idx][tgt_stm_idx])
                        
                        skip = False
                        for idx, path in last_read.items():
                            if curr_path < path: # a1->a2, a1->a2->a3
                                last_read.pop(idx)  # remove a3
                                break
                            elif curr_path > path:  # a1->a2->a3, a1->a2
                                skip = True  # skip 
                                break
                        if not skip:
                            last_read[tgt_idx] = curr_path

                if last_read:
                    src_node = node_src
                    for tgt_idx in last_read:
                        tgt_node = self.ast.get_node(tgt_idx)
                        self.add_edge(src_node, tgt_node, 'LastRead')

    def add_last_write(self):
        def check(src_stm_idx, tgt_stm_idx):
            return src_stm_idx in self.connect_stm_paris and \
                tgt_stm_idx in self.connect_stm_paris[src_stm_idx]

        for var_chain in self.registered_vars.values():
            for node_tgt in var_chain:
                last_write = {}
                tgt_stm_idx, tgt_idx = node_tgt.data['stm_idx'], node_tgt.identifier
                for node_src in var_chain:
                    src_stm_idx, src_idx = node_src.data['stm_idx'], node_src.identifier
                    if src_idx not in self.assigned_vars:
                        continue

                    if src_idx != tgt_idx and check(src_stm_idx, tgt_stm_idx):                        
                        skip = False
                        curr_path = set(self.connect_stm_paris[src_stm_idx][tgt_stm_idx])
                        for idx, path in list(last_write.items()):
                            if curr_path < path: # a2->a3, a1->a2->a3
                                last_write.pop(idx)  # remove a1
                            elif curr_path > path:  # a1->a2->a3, a2->a2
                                skip = True  # skip 
                                break
                        if not skip:
                            last_write[src_idx] = curr_path

                if last_write:
                    tgt_node = node_tgt
                    for src_idx in last_write:
                        src_node = self.ast.get_node(src_idx)
                        self.add_edge(tgt_node, src_node, 'LastWrite')
                    
    def add_compute_from(self):
        for stm_idx in self.stm_subtrees:
            stm_top_node = self.ast.get_node(stm_idx)
            if stm_top_node.tag not in ['local_variable_declaration', 'assignment_expression']:
                continue
                
            assign_idx = -1  # idx of `=`
            for n in self.stm_subtrees[stm_idx].all_nodes_itr():
                if n.tag == '=':
                    assign_idx = n.identifier
                    break

            var_nodes = self.vars_per_stm[stm_idx]
            right_nodes = list(filter(lambda n: n.identifier > assign_idx, var_nodes))
            for src_node in filter(lambda n: n.identifier < assign_idx, var_nodes):
                for tgt_node in right_nodes:
                    self.add_edge(src_node, tgt_node, 'ComputeFrom')
        
    def add_lexical_use(self):
        for var_nodes in self.registered_vars.values():
            sorted_var_idx = sorted([n.identifier for n in var_nodes])
            tgt_node = self.ast.get_node(sorted_var_idx[0])
            for i in range(1, len(sorted_var_idx)):
                src_node, tgt_node = tgt_node, self.ast.get_node(sorted_var_idx[i])
                self.add_edge(src_node, tgt_node, 'LexicalUse')
        
    def add_child_edge(self):
        for n in self.ast.all_nodes_itr():
            for c in self.ast.children(n.identifier):
                self.add_edge(n, c, 'Child')
                
    def add_next_token_edge(self):
        tok_idx = sorted([leaf.identifier for leaf in self.ast.leaves()])

        if len(tok_idx) < 2:
            return

        tgt_node = self.ast.get_node(tok_idx[0])
        for i in range(1, len(tok_idx)):
            src_node, tgt_node = tgt_node, self.ast.get_node(tok_idx[i])
            self.add_edge(src_node, tgt_node, 'NextToken')
            
    def add_leaf_cfg_edge(self):
        """ move down the cfg-edge into token level """
        def begin_leaf(stm_idx: int) -> treelib.Node:
            stm_tree = self.ast.subtree(stm_idx)
            tok_idx = sorted([leaf.identifier for leaf in stm_tree.leaves()])
            return self.ast.get_node(tok_idx[0])

        map_stm_begin_leaf = {}
        for ast_idx in self.simple_cfg.nodes:
            map_stm_begin_leaf[ast_idx] = begin_leaf(ast_idx)
        
        for u, v in self.simple_cfg.edges:
            src_node = map_stm_begin_leaf[u]
            tgt_node = map_stm_begin_leaf[v]
            self.add_edge(src_node, tgt_node, 'LeafCFG')
    
    def add_edge(self, src: treelib.Node, tgt: treelib.Node, type: str):
        """ add edge into `self.dfg` """
        self.dfg.add_edge(
            u_for_edge=src.identifier,
            v_for_edge=tgt.identifier,
            type=type,
            src=src.data['token'],
            tgt=tgt.data['token']
        )

    def variable_scope(self, stm_idx):
        """ get scope of variable in given stm(`stm_idx`)."""
        scope = []
        p = self.stm_ast.get_node(stm_idx)
        while p:
            p_idx = p.identifier
            scope.append(p_idx)
            p = self.stm_ast.parent(p_idx)
        return list(reversed(scope))

    def simplify_cfg(self):
        """ remove start and end node and replace node idf with its ast idx """
        new_cfg = nx.DiGraph()
        for (u, v) in self.cfg.edges:
            if v == 'end':
                continue
            if u == 'start':
                u = self.ast.root
                v = int(v.split('_')[-1])
            else:
                u = int(u.split('_')[-1])
                v = int(v.split('_')[-1])
            new_cfg.add_edge(u, v)
        return new_cfg

    def plot_dfg(self):
        # this plot requires pygraphviz package
        pos = nx.nx_agraph.graphviz_layout(self.dfg, prog='dot')
        nx.draw(self.dfg, pos, with_labels=False, node_size=10, 
            node_color=[[.5, .5, .5]], arrowsize=4)
        plt.show()
    
    def print_edges(self):
        for u, v, t in self.dfg.edges:
            edge = self.dfg.edges[u, v, t]
            print(f"{edge['type']}: {edge['src']}({u}) -> {edge['tgt']}({v})")
