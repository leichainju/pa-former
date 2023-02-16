""" Multi-language CFG parser
by Lei Chai, 2022-01-05
"""

from abc import ABCMeta, abstractmethod
import treelib
import networkx as nx
from yacs.config import CfgNode


class CFG(metaclass=ABCMeta):
    """cfg generation from ast based on networkx"""
    def __init__(self):
        # structure rep.
        self.ast = None
        self.cfg = None

        # exit nodes
        self.exit_idfs = []

        # stacks for state information tracking, idf -> identifier
        self.link_stack = []  # where to insert node in current hierarchy
        self.block_start_idf_stack = []  # the start node idf of expanded block
        self.meet_interrupt_stack = []  # interrupt by `break`/`continue` in block
        self.break_idfs_stack = []
        self.meet_continue_stack = []
        self.continue_idfs_stack = []
        self.default_in_switch_stack = []
        self.meet_break_stack_rcsd = []  # rcsd: repeat condition & switch & do
        self.meet_continue_stack_rcd = []  # rcd: repeat condition & do
        self.meet_break_previous_stack = []

        # patch for inner function definition
        self.inner_function_list = []
        self.inner_function_name_ast_idx = []
        self.func_calls_dict = dict()

        # init stm_config
        self.stm_config = CfgNode()
        self._set_stm_config()

    def _init_cfg(self):
        del self.cfg
        self.cfg: nx.DiGraph = nx.DiGraph()
        self.cfg.add_node('start')
        self.cfg.add_node('end')
        self.cfg.add_edge('start', 'end')

        # update link stack
        self.link_stack.append(('start', 'end'))

    def _reset_stack(self):
        self.exit_idfs = []
        self.link_stack = []
        self.block_start_idf_stack = []
        self.meet_interrupt_stack = []
        self.break_idfs_stack = []
        self.meet_continue_stack = []
        self.continue_idfs_stack = []
        self.default_in_switch_stack = []
        self.meet_break_stack_rcsd = []
        self.meet_continue_stack_rcd = []
        self.meet_break_previous_stack = []
        self._init_cfg()

        self.inner_function_list = []
        self.inner_function_name_ast_idx = []
        self.func_calls_dict = {}

    @abstractmethod
    def _set_stm_config(self):
        """some stms share the same process method."""

    @abstractmethod
    def scheduler(self, node: treelib.Node):
        """this method decides how to process `node`."""

    def gen_cfg(self, ast: treelib.Tree):
        # construct cfg
        self.ast = ast
        self._reset_stack()
        root_idx = self.ast.root

        # expand root node
        for c_node in self.sorted_node(self.ast.children(root_idx)):
            self.scheduler(c_node)

        for end_stm_idf in self.exit_idfs:
            for c_node in list(self.cfg.successors(end_stm_idf)):
                self.safely_remove_edge(end_stm_idf, c_node)
            self.cfg.add_edge(end_stm_idf, 'end')

        # inner function call
        for ast_idx in self.inner_function_name_ast_idx:
            if self.func_calls_dict.get(ast_idx, None) is not None:
                self.func_calls_dict.pop(ast_idx)
        for inner_func_name, inner_func_stm_idf in self.inner_function_list:
            for call_name, call_stm_idf in self.func_calls_dict.values():
                if call_name == inner_func_name:
                    self.cfg.add_edge(call_stm_idf, inner_func_stm_idf)

        return self.cfg

    def add_normal_statement(self, node: treelib.Node) -> str:
        """add the statement corresponding to `node`"""
        # remove old link
        pre_idf, suc_idf = self.link_stack.pop()
        self.safely_remove_edge(pre_idf, suc_idf)

        # add new node
        stm_idf = f'{node.tag}_{node.identifier}'
        self.cfg.add_node(stm_idf, ast_idx=node.identifier)

        # link the new node with predecessor(s)
        if pre_idf == suc_idf:
            this_pre_idfs = list(self.cfg.predecessors(pre_idf))
            for this_pre_idf in this_pre_idfs:
                self.cfg.add_edge(this_pre_idf, stm_idf)
                self.cfg.remove_edge(this_pre_idf, suc_idf)
        else:
            self.cfg.add_edge(pre_idf, stm_idf)

        # link the new node with successor(s)
        self.cfg.add_edge(stm_idf, suc_idf)

        if self.block_start_idf_stack[-1] is None:
            self.block_start_idf_stack[-1] = stm_idf

        for leaf in self.ast.subtree(node.identifier).leaves():
            if leaf.tag == 'identifier':
                self.func_calls_dict[leaf.identifier] = (leaf.data['token'], stm_idf)

        # next, insert nodes between `(stm_idf, suc_idf)`
        self.link_stack.append((stm_idf, suc_idf))

        return stm_idf

    def expand_node(self, node: treelib.Node, pop_start=True, ignore_interrupt=False):
        self.block_start_idf_stack.append(None)
        if not ignore_interrupt:
            self.meet_interrupt_stack.append(False)

        for c_node in self.sorted_node(self.ast.children(node.identifier)):
            self.scheduler(c_node)
            if not ignore_interrupt and self.meet_interrupt_stack[-1]:  # meet `break`/`continue`, should stop
                break

        if not ignore_interrupt:
            self.meet_interrupt_stack.pop()
        if pop_start:
            self.block_start_idf_stack.pop()

    def find_node_children(self, node, tgt_tag='block', only=True):
        """find node children with `tgt_tag`"""
        tgt_nodes = [n for n in self.ast.children(node.identifier) if n.tag == tgt_tag]
        if only:
            if len(tgt_nodes) == 1:
                return tgt_nodes[0]
            elif len(tgt_nodes) == 0:
                return None
            else:
                raise LookupError
        else:
            return tgt_nodes

    def safely_remove_edge(self, pre_idf, suc_idf):
        """safely remove cfg edge"""
        if self.cfg.has_edge(pre_idf, suc_idf):
            self.cfg.remove_edge(pre_idf, suc_idf)

    def collect_block_exit_idfs(self, link):
        """collect the exit idfs of `block` lead by de `block_pre_idf`."""
        block_exit_idf, block_end_idf = link
        assert block_end_idf == 'end', '[cfg if] conflict on end_idf'

        # get this new blocks' exit idf(s)
        if block_exit_idf == block_end_idf:  # this block has many exit link
            self.safely_remove_edge(block_end_idf, block_end_idf)
            block_exit_idfs = list(self.cfg.predecessors(block_end_idf))
        elif not self.meet_interrupt_stack[-1]:  # pass `return`/`break`/`continue`
            block_exit_idfs = [block_exit_idf]
        else:
            block_exit_idfs = []

        # remove the links of (exit, end) temporarily
        for exit_idf in block_exit_idfs:
            self.safely_remove_edge(exit_idf, block_end_idf)

        return block_exit_idfs

    @staticmethod
    def sorted_node(nodes: list):
        """sort the non-terminal nodes of ast by their identifier."""
        return filter(lambda x: not x.is_leaf(), sorted(nodes, key=lambda x: x.identifier))

    @staticmethod
    def visualize_cfg(cfg, vi_cfg_path=None):
        """visualize cfg for checking and analysis"""
        from graphviz import Digraph
        gz = Digraph(name='cfg', format='png')

        # add nodes
        for n in cfg.nodes:
            gz.node(name=n, label=n, shape='box')

        # add edges
        for (s, e) in cfg.edges:
            gz.edge(s, e)

        if vi_cfg_path is not None:
            gz.render(vi_cfg_path)

        return gz


class PythonCFG(CFG):
    """ CFG generator for Python"""
    def __init__(self):
        super(PythonCFG, self).__init__()

    def _set_stm_config(self):
        self.stm_config.normal_stm = [
            'expression_statement',
            'delete_statement',
            'assert_statement',
            'global_statement',
            'nonlocal_statement',
            'import_statement',
            'import_from_statement',
            'exec_statement']
        self.stm_config.repeat_condition_stm = [
            'for_statement',
            'while_statement']
        self.stm_config.exit_stm = [
            'return_statement',
            'raise_statement',
            'yield_statement']

    def scheduler(self, node: treelib.Node):
        if node is None:
            return
        elif node.tag in self.stm_config.normal_stm:
            self.add_normal_statement(node)
        elif node.tag in self.stm_config.repeat_condition_stm:
            self.add_repeat_condition_statement(node)
        elif node.tag in self.stm_config.exit_stm:
            self.add_exit_statement(node)
        elif node.tag == 'try_statement':
            self.add_try_statement(node)
        elif node.tag == 'if_statement':
            self.add_if_statement(node)
        elif node.tag == 'break_statement':
            self.add_break_statement(node)
        elif node.tag == 'continue_statement':
            self.add_continue_statement(node)
        elif node.tag == 'with_statement':
            self.add_with_statement(node)
        elif node.tag == 'function_definition':
            self.add_inner_function(node)
        elif node.tag == 'class_definition':
            self.add_inner_class(node)
        elif node.tag == 'decorated_definition':
            self.add_inner_decorator(node)
        elif node.tag.split('_')[-1] == 'statement':  # not-specific statement
            self.add_normal_statement(node)
            self.expand_node(node)
        else:
            self.expand_node(node)

    def simple_add_stm(self, node: treelib.Node):
        stm_idf = f'{node.tag}_{node.identifier}'
        self.cfg.add_node(stm_idf, ast_idx=node.identifier)

        for leaf in self.ast.subtree(node.identifier).leaves():
            if leaf.tag == 'identifier':
                self.func_calls_dict[leaf.identifier] = (leaf.data['token'], stm_idf)

        return stm_idf

    def add_inner_decorator(self, node: treelib.Node):
        inner_func_node = self.find_node_children(node, 'function_definition')
        inner_class_node = self.find_node_children(node, 'class_definition')
        decorator_nodes = self.find_node_children(node, 'decorator', only=False)

        if inner_func_node is not None:
            inner_def_idf = self.add_inner_function(inner_func_node)
        elif inner_class_node is not None:
            inner_def_idf = self.add_inner_class(inner_class_node)
        else:
            raise LookupError

        for decorator_node in decorator_nodes:
            decorator_idf = self.simple_add_stm(decorator_node)
            self.cfg.add_edge(inner_def_idf, decorator_idf)

    def add_inner_class(self, node: treelib.Node):
        inner_class_idf = f'{node.tag}_{node.identifier}'
        class_name_node: treelib.Node = self.find_node_children(node, 'identifier')
        class_block: treelib.Node = self.find_node_children(node, 'block')
        self.inner_function_list.append((class_name_node.data['token'], inner_class_idf))
        self.inner_function_name_ast_idx.append(class_name_node.identifier)

        expression_stm_nodes = self.find_node_children(class_block, 'expression_statement', only=False)
        function_def_nodes = self.find_node_children(class_block, 'function_definition', only=False)

        self.cfg.add_node(inner_class_idf, ast_idx=node.identifier)
        for expression_stm_node in expression_stm_nodes:
            expression_stm_idf = self.simple_add_stm(expression_stm_node)
            self.cfg.add_edge(inner_class_idf, expression_stm_idf)
        for function_def_node in function_def_nodes:
            function_def_idf = self.add_inner_function(function_def_node)
            self.cfg.add_edge(inner_class_idf, function_def_idf)

        return inner_class_idf

    def add_inner_function(self, node: treelib.Node):
        inner_func_idf = f'{node.tag}_{node.identifier}'
        func_name_node: treelib.Node = self.find_node_children(node, 'identifier')
        self.inner_function_list.append((func_name_node.data['token'], inner_func_idf))
        self.inner_function_name_ast_idx.append(func_name_node.identifier)

        inner_ast = self.ast.subtree(node.identifier)
        inner_cfg_parser = PythonCFG()
        inner_cfg: nx.DiGraph = inner_cfg_parser.gen_cfg(inner_ast)
        inner_cfg.add_node(inner_func_idf, ast_idx=node.identifier)
        for node in inner_cfg.successors('start'):
            inner_cfg.add_edge(inner_func_idf, node)
        inner_cfg.remove_node('start')
        inner_cfg.remove_node('end')

        # merger data
        self.cfg = nx.compose(self.cfg, inner_cfg)
        self.func_calls_dict.update(inner_cfg_parser.func_calls_dict)

        return inner_func_idf

    def add_repeat_condition_statement(self, node: treelib.Node):
        """add `for/while_statement`, expand `block`, `else_clause`
        and handle `break/continue`.
        """
        # update state stack
        self.meet_break_stack_rcsd.append(False)  # `break` about skip
        self.meet_continue_stack_rcd.append(False)  # `continue` about skip
        self.meet_break_previous_stack.append(False)

        # add condition_stm and expand block
        repeat_condition_stm_idf = self.add_normal_statement(node)
        self.expand_node(self.find_node_children(node))

        # add links of (block_end, repeat_condition)
        block_exit_idf, block_end_idf = self.link_stack.pop()
        assert block_end_idf == 'end', '[cfg for/while] conflict on end_idf '
        if block_exit_idf == 'end':
            self.safely_remove_edge(block_end_idf, block_end_idf)
            block_exit_idfs = list(self.cfg.predecessors(block_end_idf))
        else:
            block_exit_idfs = [block_exit_idf]

        for block_exit_idf in block_exit_idfs:
            self.cfg.add_edge(block_exit_idf, repeat_condition_stm_idf)
            self.safely_remove_edge(block_exit_idf, block_end_idf)

        # add else_clause
        else_clause = self.find_node_children(node, 'else_clause')
        if else_clause is not None:
            self.link_stack.append((repeat_condition_stm_idf, block_end_idf))
            self.add_normal_statement(else_clause)
            self.expand_node(self.find_node_children(else_clause))

        # add links of (break, end)
        if self.meet_break_stack_rcsd[-1]:
            for break_stm_idf in self.break_idfs_stack.pop():
                self.cfg.add_edge(break_stm_idf, block_end_idf)
            # if len(self.link_stack) > 0 and self.link_stack[-1][-1] != 'end':
            #     self.link_stack.pop()
            self.link_stack.append((block_end_idf, block_end_idf))
        elif else_clause is None:
            self.link_stack.append((repeat_condition_stm_idf, block_end_idf))

        # add links of (continue, end)
        if self.meet_continue_stack_rcd[-1]:
            for continue_stm_idf in self.continue_idfs_stack.pop():
                self.cfg.add_edge(continue_stm_idf, repeat_condition_stm_idf)

        # add link of (condition, end)
        if else_clause is None:
            self.cfg.add_edge(repeat_condition_stm_idf, block_end_idf)

        # update state stack
        self.meet_break_stack_rcsd.pop()
        self.meet_continue_stack_rcd.pop()
        self.meet_break_previous_stack.pop()

    def add_exit_statement(self, node: treelib.Node):
        # add `exit_statement`, unlink `stm_idf`-`suc_idf` and save this `stm_idf`
        _, suc_idf = self.link_stack[-1]
        return_stm_idf = self.add_normal_statement(node)
        self.cfg.remove_edge(return_stm_idf, suc_idf)
        self.exit_idfs.append(return_stm_idf)

    def add_try_statement(self, node: treelib.Node):
        """expand `block`(try), N x `except_clause`, `else_clause` and `finally_clause`."""
        try_block = self.find_node_children(node)
        except_clauses = self.find_node_children(node, 'except_clause', only=False)
        else_clause = self.find_node_children(node, 'else_clause')
        finally_clause = self.find_node_children(node, 'finally_clause')

        # add `try_statement` and expand try-`block`
        try_stm_idf = self.add_normal_statement(node)
        self.meet_interrupt_stack.append(False)
        self.expand_node(try_block, ignore_interrupt=True)
        try_exit_idfs = self.collect_block_exit_idfs(self.link_stack[-1])
        self.meet_interrupt_stack.pop()
        _, try_suc_idf = self.link_stack[-1]
        assert try_suc_idf == 'end', '[cfg try] conflict on `suc_idf` != end'

        # add `except_clause`s
        except_exit_idfs = []
        first_except_stm_idf = None
        for except_clause in self.sorted_node(except_clauses):
            except_stm_idf = self.add_normal_statement(except_clause)
            if first_except_stm_idf is None:
                first_except_stm_idf = except_stm_idf
            self.meet_interrupt_stack.append(False)
            self.expand_node(self.find_node_children(except_clause), ignore_interrupt=True)
            except_exit_idfs += self.collect_block_exit_idfs(self.link_stack.pop())
            self.meet_interrupt_stack.pop()
            self.link_stack.append((except_stm_idf, try_suc_idf))

        # add links of (try_idf, first_except_stm)
        if first_except_stm_idf is not None:
            self.cfg.add_edge(try_stm_idf, first_except_stm_idf)

        # add `else_clause`
        if else_clause is not None:
            self.add_normal_statement(else_clause)
            self.expand_node(self.find_node_children(else_clause))
        elif len(except_clauses) > 0:  # error not be caught or no error, go directly
            except_exit_idfs.append(self.link_stack[-1][0])

        # add `finally_clause`
        if finally_clause is not None:
            finally_stm_idf = self.add_normal_statement(finally_clause)
            self.expand_node(self.find_node_children(finally_clause))
            try_suc_idf = finally_stm_idf

        # print('try_exit_idfs', try_exit_idfs)
        # print('except_exit_idfs', except_exit_idfs)
        # add links of (`except_exit_idf`, 'try_suc_idf'/`finally_stm_idf`)
        for exit_idf in try_exit_idfs + except_exit_idfs:
            self.cfg.add_edge(exit_idf, try_suc_idf)

        _, try_suc_idf = self.link_stack.pop()
        self.link_stack.append((try_suc_idf, try_suc_idf))

    def add_if_statement(self, node: treelib.Node):
        """add `if_statement` and expand Nx`elif_clause` and `else_clause`."""
        _, suc_idf = self.link_stack[-1]
        assert suc_idf == 'end', '[cfg if] conflict on end_idf'
        if_exit_idfs = []  # this idfs directly exit from this if hierarchy

        # if_statement has: if_`block`, N x `elif_block` and {0/1} x `else_block`
        if_block = self.find_node_children(node)
        elif_blocks = self.find_node_children(node, 'elif_clause', False)
        else_block = self.find_node_children(node, 'else_clause')

        # expand if_block
        if_stm_idf = self.add_normal_statement(node)
        self.meet_interrupt_stack.append(False)
        self.expand_node(if_block, ignore_interrupt=True)
        if_exit_idfs += self.collect_block_exit_idfs(self.link_stack.pop())
        self.link_stack.append((if_stm_idf, suc_idf))
        self.meet_interrupt_stack.pop()

        # expand elif block
        for elif_block in elif_blocks:
            elif_stm_idf = self.add_normal_statement(elif_block)
            self.meet_interrupt_stack.append(False)
            self.expand_node(self.find_node_children(elif_block), ignore_interrupt=True)
            if_exit_idfs += self.collect_block_exit_idfs(self.link_stack.pop())
            self.link_stack.append((elif_stm_idf, suc_idf))
            self.meet_interrupt_stack.pop()

        # expand else block
        if else_block is not None:
            else_stm_idf = self.add_normal_statement(else_block)
            self.meet_interrupt_stack.append(False)
            self.expand_node(self.find_node_children(else_block), ignore_interrupt=True)
            if_exit_idfs += self.collect_block_exit_idfs(self.link_stack.pop())
            self.link_stack.append((else_stm_idf, suc_idf))
            self.meet_interrupt_stack.pop()
        else:
            curr_pre_idf, _ = self.link_stack[-1]
            if_exit_idfs.append(curr_pre_idf)

        # add links of (exit, end)
        for if_exit_idf in if_exit_idfs:
            self.cfg.add_edge(if_exit_idf, suc_idf)

        self.link_stack.pop()
        self.link_stack.append((suc_idf, suc_idf))

    def add_continue_statement(self, node: treelib.Node):
        """add `continue_statement`."""
        continue_stm_idf = self.add_normal_statement(node)
        pre_idf, suc_idf = self.link_stack[-1]
        self.cfg.remove_edge(pre_idf, suc_idf)

        # register continue idfs in this hierarchy
        if self.meet_continue_stack_rcd[-1]:
            self.continue_idfs_stack[-1].append(continue_stm_idf)
        else:
            self.meet_continue_stack_rcd[-1] = True
            self.continue_idfs_stack.append([continue_stm_idf])

        self.meet_interrupt_stack[-1] = True

    def add_break_statement(self, node: treelib.Node):
        """add `break_statement`."""
        self.meet_break_previous_stack[-1] = True
        break_stm_idf = self.add_normal_statement(node)
        pre_idf, suc_idf = self.link_stack[-1]
        self.cfg.remove_edge(pre_idf, suc_idf)

        # register break idfs in this hierarchy
        if self.meet_break_stack_rcsd[-1]:
            self.break_idfs_stack[-1].append(break_stm_idf)
        else:
            self.meet_break_stack_rcsd[-1] = True
            self.break_idfs_stack.append([break_stm_idf])

        self.meet_interrupt_stack[-1] = True

    def add_with_statement(self, node: treelib.Node):
        """add `with_statement` and expand `block`"""
        self.add_normal_statement(node)
        self.expand_node(self.find_node_children(node))


class JavaCFG(CFG):
    """CFG generator for Java"""
    def __init__(self):
        super(JavaCFG, self).__init__()
        self.has_switch_default_stack = []

    def _set_stm_config(self):
        self.stm_config.normal_stm = [
            'expression_statement',
            'explicit_constructor_invocation',
            'local_variable_declaration']
        self.stm_config.repeat_condition_stm = [
            'for_statement',
            'enhanced_for_statement',
            'while_statement']
        self.stm_config.exit_stm = [
            'return_statement',
            'yield_statement',
            'throw_statement']

    def scheduler(self, node: treelib.Node):
        # this method decides how to process `node`
        if node is None:
            return
        elif node.tag in self.stm_config.normal_stm:
            self.add_normal_statement(node)
        elif node.tag in self.stm_config.repeat_condition_stm:
            self.add_repeat_condition_statement(node)
        elif node.tag in self.stm_config.exit_stm:
            self.add_exit_statement(node)
        elif node.tag == 'try_statement':
            self.add_try_statement(node)
        elif node.tag == 'if_statement':
            self.add_if_statement(node)
        elif node.tag == 'switch_expression':
            self.add_switch_statement(node)
        elif node.tag == 'break_statement':
            self.add_break_statement(node)
        elif node.tag == 'do_statement':
            self.add_do_statement(node)
        elif node.tag == 'continue_statement':
            self.add_continue_statement(node)
        elif node.tag.split('_')[-1] == 'statement':  # not-specific statement
            self.add_normal_statement(node)
            self.expand_node(node)
        else:
            self.expand_node(node)

    def add_repeat_condition_statement(self, node: treelib.Node):
        """add `for/while_statement`, expand `block` and handle `break/continue`."""
        # update state stack
        self.meet_break_stack_rcsd.append(False)  # `break` about skip
        self.meet_continue_stack_rcd.append(False)  # `continue` about skip
        self.meet_break_previous_stack.append(False)

        # add condition_stm and expand block
        repeat_condition_stm_idf = self.add_normal_statement(node)
        self.meet_interrupt_stack.append(False)
        block_node = self.find_node_children(node)
        if block_node is not None:
            self.expand_node(block_node, ignore_interrupt=True)
        else:
            self.expand_node(node, ignore_interrupt=True)

        _, block_end_idf = self.link_stack[-1]
        block_exit_idfs = self.collect_block_exit_idfs(self.link_stack.pop())
        self.meet_interrupt_stack.pop()
        self.cfg.add_edge(repeat_condition_stm_idf, block_end_idf)  # add link of (condition, end)
        self.link_stack.append((repeat_condition_stm_idf, block_end_idf))

        # add links of (exit_idfs, end)
        for block_exit_idf in block_exit_idfs:
            self.cfg.add_edge(block_exit_idf, repeat_condition_stm_idf)

        # add links of (break, end)
        if self.meet_break_stack_rcsd[-1]:
            for break_stm_idf in self.break_idfs_stack.pop():
                self.cfg.add_edge(break_stm_idf, block_end_idf)
            self.link_stack.append((block_end_idf, block_end_idf))

        # add links of (continue, end)
        if self.meet_continue_stack_rcd[-1]:
            for continue_stm_idf in self.continue_idfs_stack.pop():
                self.cfg.add_edge(continue_stm_idf, repeat_condition_stm_idf)

        # update state stack
        self.meet_break_stack_rcsd.pop()
        self.meet_continue_stack_rcd.pop()
        self.meet_break_previous_stack.pop()

    def add_try_statement(self, node: treelib.Node):
        """expand `block`(try), N x `except_clause` and `finally_clause`."""
        try_block = self.find_node_children(node)
        except_clauses = self.find_node_children(node, 'catch_clause', only=False)
        finally_clause = self.find_node_children(node, 'finally_clause')

        # add `try_statement` and expand try-`block`
        try_stm_idf = self.add_normal_statement(node)
        self.expand_node(try_block, ignore_interrupt=True)
        _, try_suc_idf = self.link_stack.pop()
        assert try_suc_idf == 'end', '[cfg try] conflict on `suc_idf` != end'
        self.link_stack.append((try_stm_idf, try_suc_idf))

        # add `except_clause`s
        except_exit_idfs = []
        for except_clause in self.sorted_node(except_clauses):
            except_stm_idf = self.add_normal_statement(except_clause)
            self.meet_interrupt_stack.append(False)
            self.expand_node(self.find_node_children(except_clause), ignore_interrupt=True)
            except_exit_idfs += self.collect_block_exit_idfs(self.link_stack.pop())
            self.meet_interrupt_stack.pop()
            self.link_stack.append((except_stm_idf, try_suc_idf))

        if len(except_clauses) > 0:  # error not be caught or no error, go directly
            except_exit_idfs.append(self.link_stack[-1][0])

        # add `finally_clause`
        if finally_clause is not None:
            self.link_stack.pop()
            self.link_stack.append((try_suc_idf, try_suc_idf))
            finally_stm_idf = self.add_normal_statement(finally_clause)
            self.expand_node(self.find_node_children(finally_clause))
            try_suc_idf = finally_stm_idf

        # add links of (`except_exit_idf`, 'try_suc_idf'/`finally_stm_idf`)
        for except_exit_idf in except_exit_idfs:
            self.cfg.add_edge(except_exit_idf, try_suc_idf)

        _, suc_idf = self.link_stack.pop()
        self.link_stack.append((suc_idf, suc_idf))

    def add_if_statement(self, node: treelib.Node):
        # add `if_statement` and expand `block`
        _, suc_idf = self.link_stack[-1]
        if_stm_idf = self.add_normal_statement(node)

        else_nodes = self.find_node_children(node, 'else', False)
        if len(else_nodes) > 0:
            blocks = []
            for cn in self.ast.children(node.identifier):
                if cn.tag not in ['if', 'else', 'parenthesized_expression']:
                    blocks.append(cn)
            assert len(blocks) == 2, '[cfg if] need 2 blocks when meet else'

            pre_idf, suc_idf = self.link_stack[-1]
            this_block, else_block = blocks
            
            self.scheduler(this_block)
            this_block_exit_idfs = self.collect_block_exit_idfs(self.link_stack.pop())
            self.link_stack.append((pre_idf, suc_idf))

            self.scheduler(else_block)
            for exit_idf in this_block_exit_idfs:
                self.cfg.add_edge(exit_idf, suc_idf)

            self.link_stack.pop()
            self.link_stack.append((suc_idf, suc_idf))
        else:
            block_node = self.find_node_children(node)
            if block_node is not None:
                self.expand_node(block_node, ignore_interrupt=True)
            else:
                self.expand_node(node, ignore_interrupt=True)
            # self.expand_node(self.find_node_children(node))
            self.link_stack.pop()
            self.cfg.add_edge(if_stm_idf, suc_idf)
            self.link_stack.append((suc_idf, suc_idf))

    def add_exit_statement(self, node: treelib.Node):
        # add `exit_statement`, unlink `stm_idf`-`suc_idf` and save this `stm_idf`
        _, suc_idf = self.link_stack[-1]
        return_stm_idf = self.add_normal_statement(node)
        self.cfg.remove_edge(return_stm_idf, suc_idf)
        self.exit_idfs.append(return_stm_idf)

    def add_break_statement(self, node: treelib.Node):
        """add `break_statement`."""
        self.meet_break_previous_stack[-1] = True
        break_stm_idf = self.add_normal_statement(node)
        pre_idf, suc_idf = self.link_stack[-1]
        self.cfg.remove_edge(pre_idf, suc_idf)

        # register break idfs in this hierarchy
        if self.meet_break_stack_rcsd[-1]:
            self.break_idfs_stack[-1].append(break_stm_idf)
        else:
            self.meet_break_stack_rcsd[-1] = True
            self.break_idfs_stack.append([break_stm_idf])

        self.meet_interrupt_stack[-1] = True

    def add_continue_statement(self, node: treelib.Node):
        """add `continue_statement`."""
        continue_stm_idf = self.add_normal_statement(node)
        pre_idf, suc_idf = self.link_stack[-1]
        self.cfg.remove_edge(pre_idf, suc_idf)

        # register continue idfs in this hierarchy
        if self.meet_continue_stack_rcd[-1]:
            self.continue_idfs_stack[-1].append(continue_stm_idf)
        else:
            self.meet_continue_stack_rcd[-1] = True
            self.continue_idfs_stack.append([continue_stm_idf])

        self.meet_interrupt_stack[-1] = True

    def add_switch_statement(self, node: treelib.Node):
        """add `switch_statement` and expand `switch_block` in parallel"""
        self.meet_break_stack_rcsd.append(False)
        self.meet_break_previous_stack.append(False)
        self.has_switch_default_stack.append(False)

        # add `switch_statement`
        self.add_normal_statement(node)

        # add `case`s
        _, suc_idf = self.link_stack[-1]
        least_case_stm_idf = None
        switch_block = self.find_node_children(node, tgt_tag='switch_block')
        for c_block in self.sorted_node(self.ast.children(switch_block.identifier)):
            # update break state
            meet_break = self.meet_break_previous_stack[-1]
            self.meet_break_previous_stack[-1] = False

            # add `case_stm` and expand `block`
            case_stm_idf = self.add_normal_statement(c_block)
            self.expand_node(c_block, pop_start=False)
            self.has_switch_default_stack[-1] |= self.meet_default(c_block)

            # meet `break` in previous `case_clause`
            if self.meet_break_previous_stack[-1]:
                self.link_stack.pop()
                self.link_stack.append((case_stm_idf, suc_idf))

            # no `break` in previous case
            if not meet_break and least_case_stm_idf is not None:
                # add link of (previous_case_block_exit, this_case_block_start)
                pre_case_block_exit_idfs = list(self.cfg.predecessors(case_stm_idf))
                this_case_block_start_idf = self.block_start_idf_stack.pop()

                for exit_idf in pre_case_block_exit_idfs:
                    self.safely_remove_edge(exit_idf, case_stm_idf)
                    if this_case_block_start_idf is not None:
                        self.cfg.add_edge(exit_idf, this_case_block_start_idf)

                # add link of (previous_case_stm, this_case_stm)
                self.cfg.add_edge(least_case_stm_idf, case_stm_idf)

            least_case_stm_idf = case_stm_idf

        if not self.has_switch_default_stack.pop():
            self.cfg.add_edge(least_case_stm_idf, suc_idf)

        if self.meet_break_stack_rcsd[-1]:
            for break_stm_idf in self.break_idfs_stack.pop():
                self.cfg.add_edge(break_stm_idf, suc_idf)
            self.link_stack.append((suc_idf, suc_idf))

        self.link_stack.pop()  # remove switch block
        self.link_stack.append((suc_idf, suc_idf))  # new start

        self.meet_break_stack_rcsd.pop()
        self.meet_break_previous_stack.pop()

    def add_do_statement(self, node: treelib.Node):
        """expand `block`, add `while_statement` and handle `break/continue`."""
        # update state stack
        self.meet_break_stack_rcsd.append(False)  # `break` about skip
        self.meet_continue_stack_rcd.append(False)  # `continue` about skip
        self.meet_break_previous_stack.append(False)

        # expand `block`
        do_block = self.find_node_children(node)
        if do_block is not None:
            self.expand_node(self.find_node_children(node), pop_start=False)
            do_block_start_idf = self.block_start_idf_stack.pop()
        else:
            non_terminal = [c for c in self.ast.children(node.identifier)
                            if not c.is_leaf() and c.tag != 'parenthesized_expression']
            if len(non_terminal) == 1:
                do_block_start_idf = self.add_normal_statement(non_terminal[0])
            elif len(non_terminal) == 0:
                do_block_start_idf = None
            else:
                raise RuntimeError

        # add `do_statement`
        do_stm_idf = self.add_normal_statement(node)
        if do_block_start_idf is None:
            do_block_start_idf = do_stm_idf
        self.cfg.add_edge(do_stm_idf, do_block_start_idf)

        _, suc_idf = self.link_stack[-1]
        if self.meet_break_stack_rcsd[-1]:
            for break_stm_idf in self.break_idfs_stack.pop():
                self.cfg.add_edge(break_stm_idf, suc_idf)
            self.link_stack.append((suc_idf, suc_idf))

        if self.meet_continue_stack_rcd[-1]:
            for continue_stm_idf in self.continue_idfs_stack.pop():
                self.cfg.add_edge(continue_stm_idf, do_stm_idf)

        self.meet_break_stack_rcsd.pop()
        self.meet_continue_stack_rcd.pop()
        self.meet_break_previous_stack.pop()

    def meet_default(self, node: treelib.Node):
        """check if this node is a `switch_default`."""
        switch_label = self.find_node_children(node, tgt_tag='switch_label')
        assert switch_label is not None
        default = self.find_node_children(switch_label, tgt_tag='default')
        return default is not None
