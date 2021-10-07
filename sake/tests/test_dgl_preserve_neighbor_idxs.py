import pytest

def test_neighbor_idxs():
    import dgl
    import networkx as nx
    g = dgl.from_networkx(nx.complete_graph(5))

    def message_func(edges):
        return {'old_idxs': edges.edges()[2]}

    def reduce_func(nodes):
        return {'old_idxs': nodes.mailbox['old_idxs']}

    g.update_all(message_func, reduce_func)

    in_edges = g.in_edges(v=[0, 1, 2, 3, 4], form='eid')

    assert (g.ndata['old_idxs'].flatten() == in_edges).all()
