import atexit
import socket

import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.distributed import LocalFeatureStore, LocalGraphStore
from torch_geometric.distributed.dist_context import DistContext
from torch_geometric.distributed.dist_neighbor_sampler import (
    DistNeighborSampler,
    close_sampler,
)
from torch_geometric.distributed.rpc import init_rpc
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput
from torch_geometric.sampler.neighbor_sampler import node_sample
from torch_geometric.testing import withPackage


def create_data(rank, world_size, temporal=False):
    num_nodes = 10
    # create dist data
    if rank == 0:
        # partition 0
        node_id = torch.tensor([0, 1, 2, 3, 4, 5, 6])
        # sorted by dst
        edge_index = torch.tensor([
            [1, 2, 3, 4, 5, 0, 0],
            [0, 1, 2, 3, 4, 4, 9],
        ])
    else:
        # partition 1
        node_id = torch.tensor([0, 4, 5, 6, 7, 8, 9])
        # sorted by dst
        edge_index = torch.tensor([
            [5, 6, 7, 8, 9, 5, 0],
            [4, 5, 6, 7, 8, 9, 9],
        ])

    feature_store = LocalFeatureStore.from_data(node_id)
    graph_store = LocalGraphStore.from_data(None, edge_index,
                                            num_nodes=num_nodes,
                                            is_sorted=True)

    graph_store.node_pb = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    graph_store.meta.update({'num_parts': 2})
    graph_store.partition_idx = rank
    graph_store.num_partitions = world_size

    dist_data = (feature_store, graph_store)

    # create reference data sorted by dst
    edge_index = torch.tensor([
        [1, 2, 3, 4, 5, 0, 5, 6, 7, 8, 9, 0],
        [0, 1, 2, 3, 4, 4, 9, 5, 6, 7, 8, 9],
    ])
    data = Data(x=None, y=None, edge_index=edge_index, num_nodes=num_nodes)

    if temporal:
        # create time data sorted by edge_index srcs
        data_time = torch.tensor([5, 0, 1, 3, 3, 4, 4, 4, 4, 4])
        feature_store.put_tensor(data_time, group_name=None, attr_name="time")

        data.time = data_time

    return (dist_data, data)


def dist_neighbor_sampler(
    world_size: int,
    rank: int,
    master_port: int,
    disjoint: bool = False,
):
    dist_data, data = create_data(rank, world_size)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name="dist-sampler-test",
    )

    # Initialize training process group of PyTorch.
    torch.distributed.init_process_group(
        backend="gloo",
        rank=current_ctx.rank,
        world_size=current_ctx.world_size,
        init_method="tcp://{}:{}".format('localhost', master_port),
    )

    num_neighbors = [-1, -1]
    dist_sampler = DistNeighborSampler(
        data=dist_data,
        current_ctx=current_ctx,
        rpc_worker_names={},
        num_neighbors=num_neighbors,
        shuffle=False,
        disjoint=disjoint,
    )

    init_rpc(
        current_ctx=current_ctx,
        rpc_worker_names={},
        master_addr='localhost',
        master_port=master_port,
    )

    dist_sampler.register_sampler_rpc()
    dist_sampler.init_event_loop()

    # close RPC & worker group at exit:
    atexit.register(close_sampler, 0, dist_sampler)
    torch.distributed.barrier()

    # seed nodes
    if rank == 0:
        input_node = torch.tensor([1, 6], dtype=torch.int64)
    else:
        input_node = torch.tensor([4, 9], dtype=torch.int64)

    inputs = NodeSamplerInput(
        input_id=None,
        node=input_node,
    )

    # evaluate distributed node sample function
    out_dist = dist_sampler.event_loop.run_task(
        coro=dist_sampler.node_sample(inputs))

    torch.distributed.barrier()

    sampler = NeighborSampler(data=data, num_neighbors=num_neighbors,
                              disjoint=disjoint)

    # evaluate node sample function
    out = node_sample(inputs, sampler._sample)

    # compare distributed output with single machine output
    assert torch.equal(out_dist.node, out.node)
    assert torch.equal(out_dist.row, out.row)
    assert torch.equal(out_dist.col, out.col)
    if disjoint:
        assert torch.equal(out_dist.batch, out.batch)
    assert out_dist.num_sampled_nodes == out.num_sampled_nodes
    assert out_dist.num_sampled_edges == out.num_sampled_edges

    torch.distributed.barrier()


def dist_neighbor_sampler_temporal(
    world_size: int,
    rank: int,
    master_port: int,
    seed_time: torch.tensor = None,
    temporal_strategy: str = 'uniform',
):
    dist_data, data = create_data(rank, world_size, temporal=True)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name="dist-sampler-test",
    )

    # Initialize training process group of PyTorch.
    torch.distributed.init_process_group(
        backend="gloo",
        rank=current_ctx.rank,
        world_size=current_ctx.world_size,
        init_method="tcp://{}:{}".format('localhost', master_port),
    )

    num_neighbors = [-1, -1] if temporal_strategy == 'uniform' else [1, 1]
    dist_sampler = DistNeighborSampler(
        data=dist_data,
        current_ctx=current_ctx,
        rpc_worker_names={},
        num_neighbors=num_neighbors,
        shuffle=False,
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr='time',
    )

    init_rpc(
        current_ctx=current_ctx,
        rpc_worker_names={},
        master_addr='localhost',
        master_port=master_port,
    )

    dist_sampler.register_sampler_rpc()
    dist_sampler.init_event_loop()

    # close RPC & worker group at exit:
    atexit.register(close_sampler, 0, dist_sampler)
    torch.distributed.barrier()

    # seed nodes
    if rank == 0:
        input_node = torch.tensor([1, 6], dtype=torch.int64)
    else:
        input_node = torch.tensor([4, 9], dtype=torch.int64)

    inputs = NodeSamplerInput(
        input_id=None,
        node=input_node,
        time=seed_time,
    )

    # evaluate distributed node sample function
    out_dist = dist_sampler.event_loop.run_task(
        coro=dist_sampler.node_sample(inputs))

    torch.distributed.barrier()

    sampler = NeighborSampler(data=data, num_neighbors=num_neighbors,
                              disjoint=True,
                              temporal_strategy=temporal_strategy,
                              time_attr='time')

    # evaluate node sample function
    out = node_sample(inputs, sampler._sample)

    # compare distributed output with single machine output
    assert torch.equal(out_dist.node, out.node)
    assert torch.equal(out_dist.row, out.row)
    assert torch.equal(out_dist.col, out.col)
    assert torch.equal(out_dist.batch, out.batch)
    assert out_dist.num_sampled_nodes == out.num_sampled_nodes
    assert out_dist.num_sampled_edges == out.num_sampled_edges

    torch.distributed.barrier()


def create_hetero_data(rank, world_size):
    if rank == 0:
        node_id_dict = {
            'paper': torch.tensor([0, 1, 2, 9], dtype=torch.int64),
            'author': torch.tensor([3, 4, 5], dtype=torch.int64),
        }
        x_dict = {'paper': torch.randn(4, 6), 'author': torch.randn(3, 6)}
        y_dict = {
            'paper': torch.randint(0, 2, (4, )),
            'author': torch.randint(0, 2, (3, ))
        }

        edge_index_dict = {
            ('paper', 'to', 'paper'): torch.tensor([
                [1, 2, 0],
                [0, 1, 9],
            ]),
            ('paper', 'to', 'author'): torch.tensor([
                [0],
                [4],
            ]),
            ('author', 'to', 'paper'): torch.tensor([
                [3],
                [2],
            ]),
            ('author', 'to', 'author'): torch.tensor([
                [4, 5],
                [3, 4],
            ])
        }
        edge_id_dict = {
            ('paper', 'to', 'paper'): torch.tensor([0, 1, 11]),
            ('paper', 'to', 'author'): torch.tensor([2]),
            ('author', 'to', 'paper'): torch.tensor([4]),
            ('author', 'to', 'author'): torch.tensor([3, 5]),
        }
    else:
        node_id_dict = {
            'paper': torch.tensor([0, 8, 9], dtype=torch.int64),
            'author': torch.tensor([4, 5, 6, 7], dtype=torch.int64),
        }
        num_nodes_dict = {'paper': 3, 'author': 4}
        x_dict = {'paper': torch.randn(3, 6), 'author': torch.randn(4, 6)}
        y_dict = {
            'paper': torch.randint(0, 2, (3, )),
            'author': torch.randint(0, 2, (4, ))
        }

        edge_index_dict = {
            ('paper', 'to', 'paper'): torch.tensor([
                [9, 0],
                [8, 9],
            ]),
            ('paper', 'to', 'author'): torch.tensor([
                [8],
                [7],
            ]),
            ('author', 'to', 'paper'): torch.tensor([
                [5],
                [9],
            ]),
            ('author', 'to', 'author'): torch.tensor([
                [5, 6, 7],
                [4, 5, 6],
            ])
        }
        edge_id_dict = {
            ('paper', 'to', 'paper'): torch.tensor([9, 11]),
            ('paper', 'to', 'author'): torch.tensor([10]),
            ('author', 'to', 'paper'): torch.tensor([8]),
            ('author', 'to', 'author'): torch.tensor([5, 6, 7]),
        }
    num_nodes_dict = {'paper': 5, 'author': 5}
    is_sorted_dict = {'paper': True, 'author': True}

    feature_store = LocalFeatureStore.from_hetero_data(node_id_dict, x_dict,
                                                       y_dict, edge_id_dict)
    graph_store = LocalGraphStore.from_hetero_data(edge_id_dict,
                                                   edge_index_dict,
                                                   num_nodes_dict,
                                                   is_sorted_dict)

    graph_store.node_pb = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    graph_store.meta.update({'num_parts': 2})
    graph_store.partition_idx = rank
    graph_store.num_partitions = world_size

    dist_data = (feature_store, graph_store)

    # create reference data
    edge_index_dict = {
        ('paper', 'to', 'paper'): torch.tensor([
            [1, 2, 9, 0],
            [0, 1, 8, 9],
        ]),
        ('paper', 'to', 'author'): torch.tensor([
            [0, 8],
            [4, 7],
        ]),
        ('author', 'to', 'paper'): torch.tensor([
            [3, 5],
            [2, 9],
        ]),
        ('author', 'to', 'author'): torch.tensor([
            [4, 5, 6, 7],
            [3, 4, 5, 6],
        ])
    }

    data = HeteroData()
    data['paper'].x = None
    data['author'].x = None
    data['paper'].y = None
    data['author'].y = None
    data[('paper', 'to', 'paper')].edge_index = edge_index_dict[('paper', 'to',
                                                                 'paper')]
    data[('paper', 'to',
          'author')].edge_index = edge_index_dict[('paper', 'to', 'author')]
    data[('author', 'to',
          'paper')].edge_index = edge_index_dict[('author', 'to', 'paper')]
    data[('author', 'to',
          'author')].edge_index = edge_index_dict[('author', 'to', 'author')]
    data['paper'].num_nodes = 10
    data['author'].num_nodes = 10

    return (dist_data, data)


def dist_neighbor_sampler_hetero(
    world_size: int,
    rank: int,
    master_port: int,
    input_type: str,
    disjoint: bool = False,
):
    dist_data, data = create_hetero_data(rank, world_size)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name="dist-sampler-test",
    )

    # Initialize training process group of PyTorch.
    torch.distributed.init_process_group(
        backend="gloo",
        rank=current_ctx.rank,
        world_size=current_ctx.world_size,
        init_method="tcp://{}:{}".format('localhost', master_port),
    )

    num_neighbors = [-1, -1]
    dist_sampler = DistNeighborSampler(
        data=dist_data,
        current_ctx=current_ctx,
        rpc_worker_names={},
        num_neighbors=num_neighbors,
        shuffle=False,
        disjoint=disjoint,
    )

    init_rpc(
        current_ctx=current_ctx,
        rpc_worker_names={},
        master_addr='localhost',
        master_port=master_port,
    )

    dist_sampler.register_sampler_rpc()
    dist_sampler.init_event_loop()

    # close RPC & worker group at exit:
    atexit.register(close_sampler, 0, dist_sampler)
    torch.distributed.barrier()

    # seed nodes
    if rank == 0:
        input_node = torch.tensor([0, 2], dtype=torch.int64)
    else:
        input_node = torch.tensor([5, 7], dtype=torch.int64)

    inputs = NodeSamplerInput(
        input_id=None,
        node=input_node,
        input_type=input_type,
    )

    # evaluate distributed node sample function
    out_dist = dist_sampler.event_loop.run_task(
        coro=dist_sampler.node_sample(inputs))

    sampler = NeighborSampler(data=data, num_neighbors=num_neighbors,
                              disjoint=disjoint, is_sorted=True)

    # evaluate node sample function
    out = node_sample(inputs, sampler._sample)

    # compare distributed output with single machine output
    for k in data.node_types:
        assert torch.equal(out_dist.node[k], out.node[k])
        if disjoint:
            assert torch.equal(out_dist.batch, out.batch)
        assert out_dist.num_sampled_nodes[k] == out.num_sampled_nodes[k]

    for k in data.edge_types:
        assert torch.equal(out_dist.row[k], out.row[k])
        assert torch.equal(out_dist.col[k], out.col[k])
        assert out_dist.num_sampled_edges[k] == out.num_sampled_edges[k]

    torch.distributed.barrier()


@withPackage('pyg_lib')
@pytest.mark.parametrize("disjoint", [False, True])
def test_dist_neighbor_sampler(disjoint):
    mp_context = torch.multiprocessing.get_context("spawn")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    world_size = 2
    w0 = mp_context.Process(
        target=dist_neighbor_sampler,
        args=(world_size, 0, port, disjoint),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler,
        args=(world_size, 1, port, disjoint),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@withPackage('pyg_lib')
@pytest.mark.parametrize("seed_time", [None, torch.tensor([3, 6])])
@pytest.mark.parametrize("temporal_strategy", ['uniform', 'last'])
def test_dist_neighbor_sampler_temporal(seed_time, temporal_strategy):
    mp_context = torch.multiprocessing.get_context("spawn")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    world_size = 2
    w0 = mp_context.Process(
        target=dist_neighbor_sampler_temporal,
        args=(world_size, 0, port, seed_time, temporal_strategy),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler_temporal,
        args=(world_size, 1, port, seed_time, temporal_strategy),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@withPackage('pyg_lib')
# @pytest.mark.parametrize("disjoint", [True, False])
def test_dist_neighbor_sampler_hetero():
    mp_context = torch.multiprocessing.get_context("spawn")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    disjoint = False
    world_size = 2
    w0 = mp_context.Process(
        target=dist_neighbor_sampler_hetero,
        args=(world_size, 0, port, 'paper', disjoint),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler_hetero,
        args=(world_size, 1, port, 'author', disjoint),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()
