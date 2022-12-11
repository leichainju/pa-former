""" sparse opeartions mainly for edge """


import torch


def gen_scatter_idx(edge_pos, seq_len):
    """ 
    Args:
        edge_pos (LongTensor): shape as `[num_edge, 2]`, the `src_idx` and `tgt_idx` of each edge.
        seq_len (int): length of this seq
    Return:
        (torch.sparse_matrix): idx map for scatter, shape as `[seq_len, seq_len + 1]`

    Example:
    >>> edge = torch.tensor([0, 1, 4, 5])  # 0 is `pad` value
    >>> pos = torch.tensor([[1, 0], [0, 0], [1, 1]])
    >>> seq_len = 3
    >>> gen_scatter_idx(pos, seq_len).to_dense()
    >>> tensor([[0, 0, 1, 0],
                [0, 1, 0, 2],
                [0, 0, 0, 0]])
    """
    v = edge_pos[:, 1] + 1
    edge_pos[:, 1] = torch.arange(1, edge_pos.size(0) + 1)
    return torch.sparse_coo_tensor(edge_pos.T, v, (seq_len, seq_len + 1))


def link_map_list(a2b_map, b2c_map):
    a2c_map = [b2c_map[i] for i in a2b_map]
    return a2c_map