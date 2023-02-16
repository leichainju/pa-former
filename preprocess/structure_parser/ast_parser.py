# Multi-language AST parser
# By Lei Chai, 2022-01-06

import treelib
import tree_sitter


class AST(object):
    """ ast operation based on treelib"""
    def __init__(self, language):
        self.language = language
        self.ast_parser = tree_sitter.Parser()
        self.ast_parser.set_language(tree_sitter.Language('./mmlang.so', language))

    def gen_ast(self, code_bytes: bytes):
        raw_ast_root = self.ast_parser.parse(code_bytes).root_node
        ast = treelib.Tree()
        root_idx = self.add_tree_node(ast, raw_ast_root, None, '')
        self.dfs(raw_ast_root, ast, root_idx, code_bytes)
        ast = self.remove_string_comment(ast)
        return self.split_class_outer(ast)

    def gen_raw_ast(self, code_bytes: bytes):
        raw_ast_root = self.ast_parser.parse(code_bytes).root_node
        ast = treelib.Tree()
        root_idx = self.add_tree_node(ast, raw_ast_root, None, '')
        self.dfs(raw_ast_root, ast, root_idx, code_bytes)
        ast = self.remove_string_comment(ast)
        return ast

    @staticmethod
    def split_class_outer(ast: treelib.Tree):
        # remove the fake class outer
        m_idx = -1
        for n in ast.all_nodes_itr():
            if n.tag in ['method_declaration', 'constructor_declaration', 'function_definition']:
                m_idx = n.identifier
                break
        if m_idx != -1:
            return ast.subtree(m_idx)
        else:
            return ast

    def remove_string_comment(self, ast: treelib.Tree):
        """comment: expression_statement-string"""
        comment_idxs = []
        for n in ast.all_nodes_itr():
            if self.language == 'python' and n.tag == 'expression_statement':
                n_children = ast.children(n.identifier)
                if len(n_children) == 1 and n_children[0].tag == 'string':
                    comment_idxs.append(n.identifier)
            elif n.tag == 'comment':
                comment_idxs.append(n.identifier)

        for comment_idx in comment_idxs:
            ast.remove_node(comment_idx)

        return ast

    @classmethod
    def dfs(cls, subtree_node: tree_sitter.Node, ast: treelib.Tree, parent_idx, b_code):
        # add terminal node
        if subtree_node.child_count == 0 or subtree_node.type in ['string']:
            b_token = b_code[subtree_node.start_byte: subtree_node.end_byte]
            token = bytes.decode(b_token)
            cls.add_tree_node(ast, subtree_node, parent_idx, token)
            return

        # add non-terminal node and skip ERROR node
        if subtree_node.type != 'ERROR':
            this_idx = cls.add_tree_node(ast, subtree_node, parent_idx, '')
        else:
            this_idx = parent_idx

        # add children of non-terminal node
        for c_node in subtree_node.children:
            cls.dfs(c_node, ast, this_idx, b_code)

    @staticmethod
    def add_tree_node(ast: treelib.Tree, node: tree_sitter.Node, parent_idx, token):
        node_idx = len(ast.all_nodes())
        ast.create_node(
            tag=node.type,
            identifier=node_idx,
            parent=parent_idx,
            data={
                'start_byte': node.start_byte,
                'end_byte': node.end_byte,
                'start_point': node.start_point,
                'token': token
            }
        )
        return node_idx
