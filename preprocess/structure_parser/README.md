# Notes for structure parser

## Schedule

- [x] generate AST
- [x] CFG for Java
- [x] CFG for Python
- [ ] ~~generate DDG~~

## Generate AST

We use the `tree_sitter` for `ast` parse. And we convert the parsed tree into `treelib.Tree` 
for efficient tree operations.

## Generate CFG

We implement the CFG generator by ourselves and this part shows the details. We use `networkx.DiGraph`
to representation the CFG for efficient graph operations.

### CFG for Java

- [x] normal statement
  - [x] expression_statement
  - [x] local_variable_declaration
- [x] for_statement
  - [x] enhanced_for / while
  - [x] break
  - [x] continue
- [x] if_statement
  - [x] else
- [x] try_statement
  - [x] catch
  - [x] finally
- [x] switch_expression
  - [x] case
  - [x] default
  - [x] break
- [x] yield_statement
- [x] return_statement
- [x] throw_statement

### CFG for Python

- [x] normal
  - [x] expression_statement
  - [x] delete_statement
  - [x] assert_statement
  - [x] nonlocal_statement
  - [x] global_statement
  - [x] exec_statement
- [x] if_statement
  - [x] elif_clause
  - [x] else_clause
- [x] repeat_condition_statement
  - [x] for_statement
  - [x] while_statement
  - [x] else_clause
  - [x] break_statement
  - [x] continue_statement
- [x] exit_statement
  - [x] raise_statement
  - [x] return_statement
- [x] try_statement
  - [x] except_clause
  - [x] else_clause
  - [x] finally_clause
- [x] comment
  - [x] block-string
- [x] with_statement
