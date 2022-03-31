import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .gnn_module import GNN
from .transformer_module import CTransformerEncoder


# Hidden Complementary Factors Learning中的一个部分图网络的整体更新过程（也就是一个分支）
class GABlock(nn.Module):
    def __init__(self, node_dim, gnn_deep, attention_head, attention_layer, attention_feed_dim):
        super(GABlock, self).__init__()
        self.node_dim = node_dim
        self.gnn_deep = gnn_deep
        self.output_dim = node_dim * 2
        self.dropout = 0.0

        # build layers
        self.gnn = self._make_gnn_module()
        self.attention = CTransformerEncoder(self.output_dim, attention_head, attention_feed_dim,
                                             attention_layer, dropout=self.dropout)

    def _make_gnn_module(self):
        gnn = nn.ModuleList()
        for i in range(self.gnn_deep):
            gnn.append(GNN(self.node_dim, bias=True, penalty_l1=True, normalize=False))
            gnn.append(nn.LeakyReLU(0.01))
            if i != self.gnn_deep - 1:
                gnn.append(nn.Dropout(p=self.dropout))
        return gnn

    # 对 graphs 计算出需补 0 的位置
    @staticmethod
    def _calc_src_key_padding_mask(graphs, is_bool=True):
        max_len = max([s.size(0) for s in graphs])
        padding_mask = torch.ones(len(graphs), max_len)
        for i, graph in enumerate(graphs):
            index = torch.tensor([max_len - ti for ti in range(1, max_len - graph.size(0) + 1)])
            if len(index):
                padding_mask[i].index_fill_(0, index, 0)  # 在 0 维度用 0 填充 index 位置的 tensor
        if is_bool:
            return (1 - padding_mask).bool()
        else:
            return padding_mask

    # 获取更新后的图和损失
    def forward(self, data):
        r"""
        preprocessed data
        :param data: type of `torch_geometric.data.Data`
        :return:  pooled node of a graph, loss
        """
        before_gnn = data.x.clone()  # (N, D)
        gnn_out = data
        gnn_type_mask_norm_container = []
        for idx, gnn in enumerate(self.gnn):
            if idx and isinstance(gnn, GNN):
                data.x = gnn_out
                gnn_out = gnn(data)
            else:
                gnn_out = gnn(gnn_out)
            if isinstance(gnn, GNN):
                gnn_type_mask_norm_container.append(gnn.embedding_l1)

        # L1损失
        type_mask_norm = torch.stack(gnn_type_mask_norm_container).mean()

        # 对应原论文，和原始的图要做一次连接
        gnn_out = torch.cat([before_gnn, gnn_out], dim=-1)  # (N, 2*D)
        # print(f"gnn_out: {gnn_out.shape}")

        # convert data structure for attention layer
        # 将图数据处理成可以输入到注意力机制的形式
        graphs = []

        # data.num_graphs = data.y = graph_N = batch_size * (num_negative + 1)
        # 若batch_size=16，num_negative=2, 则graph_N为48
        # len(data.slices_indicator) = data.num_graphs + 1

        for shift in range(data.num_graphs):
            tmp_slices = data.slices_indicator[shift:shift + 2]
            graphs.append(gnn_out[tmp_slices[0].item():tmp_slices[1].item()])

        # 2 * batch_size = Graph_N
        batch_data = pad_sequence(graphs)  # (seq,Graph_N,E), 'seq' is length of the longest sequence in graphs.

        # print(f'batch_data: {batch_data.shape}')

        padding_mask = self._calc_src_key_padding_mask(graphs).to(gnn_out.device)

        output, final_att = self.attention(batch_data, padding_mask)  # (S,Graph_N,E), (Graph_N,S), S=seq
        output.transpose_(0, 1)  # (Graph_N,S,E), E = 2D, D为embedding

        # print(f'branch_model output: {output.shape}')

        # noinspection PyTypeChecker
        output = torch.einsum('bij,bi->bij', [output, final_att])  # (Graph_N,S,E)
        gather = output.sum(1)  # (Graph_N,E)

        # print(f'gather: {gather.shape}')

        return gather, type_mask_norm
