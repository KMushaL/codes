import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .branch_module import GABlock
from .resnet_18 import resnet18


# 获得一个 branch 的评分
class BranchScore(nn.Module):
    def __init__(self, **kwargs):
        super(BranchScore, self).__init__()
        self.trans_w = nn.Linear(kwargs["node_dim"], kwargs["node_dim"], bias=False)
        self.attr_gablock = GABlock(**kwargs)
        self.attr_score = nn.Linear(self.attr_gablock.output_dim, 1)

        self.reset_parameters()

    @staticmethod
    def _init_sequential(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, 0, 0.001)
            nn.init.constant_(m.bias, 0)

    def reset_parameters(self):
        self.attr_score.apply(self._init_sequential)

    def forward(self, data):
        img = data.x
        img2attr = self.trans_w(img)
        data.x = img2attr
        # data.x = img2attr: [一个batch的seman_category类别数, image_embedding 即 D]

        # print(f"data: {len(data)}, data.x: {data.x.shape}",
        #       f"data.rcid_index: {data.rcid_index.shape}")

        attr, type_mask_norm = self.attr_gablock(data)  # attr: (Graph_N, 2 * D)
        score_attr = self.attr_score(attr)  # 一个branch提取的整体服装特征，维度为(Graph_N, 1)

        # print(f"attr: {attr.shape}, score_attr: {score_attr.shape}")

        return img2attr, attr, score_attr, type_mask_norm


class CORL(nn.Module):
    """
    将所有branch算出的评分进行汇总并返回
    """
    def __init__(self, attr_aspect, **kwargs):
        super(CORL, self).__init__()
        self.attr_aspect = attr_aspect
        self.attr_branches = self._make_branch(attr_aspect, **kwargs)

    @staticmethod
    def _make_branch(attr_aspect, **kwargs):
        """
        attr_aspect: 分支数
        """
        branches = nn.ModuleList()
        for branch in range(attr_aspect):
            branches.append(BranchScore(**kwargs))
        return branches

    def forward(self, data):
        img_node = data.x.clone()  # (N,D) N为一个batch（包括正样本和负样本）的img_item数,D 为embedding

        # print(f"img_node: {img_node.shape}")

        img2attrs, attrs, score_attrs, type_mask_norms = [], [], [], []

        for branch in self.attr_branches:
            data.x = img_node

            # score_attr: (Graph_N, 1) 为服装的全局特征

            img2attr, attr, score_attr, type_mask_norm = branch(data)  # (N,D),(Graph_N,2 * D),(Graph_N,1),(1)
            # print(f"img2attr: {img2attr.shape}, attr: {attr.shape}, score_attr: {score_attr.shape}")

            img2attrs.append(img2attr)
            attrs.append(attr)
            score_attrs.append(score_attr)  # score_attrs: 共branch个(Graph_N, 1)的tensor
            type_mask_norms.append(type_mask_norm)

        # complementarity regularization loss
        mat_feats = F.normalize(torch.stack(img2attrs, dim=1), p=2, dim=-1)  # N,branch,D
        diversity_mat = torch.bmm(mat_feats, mat_feats.transpose(1, 2))  # N,branch,branch
        eye_mat = torch.eye(diversity_mat.size(-1)).unsqueeze(0).repeat(diversity_mat.size(0), 1, 1).to(
            diversity_mat.device)
        diversity_loss = torch.pow(eye_mat - diversity_mat, exponent=2).sum()

        # tot_type_mask_norm: 标量
        tot_type_mask_norm = torch.stack(type_mask_norms).mean()

        # torch.stack(score_attrs, dim=1).shape: (Graph_N, branch, 1)
        # fine score: (Graph_N, 1) 最终的评分
        fine_score = torch.stack(score_attrs, dim=1).sum(dim=1)
        # print(f"score_attrs: {torch.stack(score_attrs, dim=1).shape}, fine_score: {fine_score.shape}")

        # fine_feature
        # 将所有branch获得的评分拼接起来，再做softmax
        # torch.cat(score_attrs, dim=-1).shape: (Graph_N, branch)
        attr_weight = F.softmax(torch.cat(score_attrs, dim=-1), dim=-1)  # (Graph_N,branch)
        fine_feature = torch.bmm(attr_weight.unsqueeze(1), torch.stack(attrs, dim=1)).squeeze(1)  # (Graph_N,2D)
        # print(f"attr_weight: {attr_weight.shape}")
        return fine_feature, fine_score, tot_type_mask_norm, diversity_loss


class CF(nn.Module):
    def __init__(self, num_node_features: int, num_cotpye: int, depth: int, nhead: int,
                 dim_feedforward: int, num_layers: int, num_category: int, num_branch: int = 5):
        super(CF, self).__init__()
        # self.cotpye_embedding = nn.Embedding(num_cotpye + 1, num_node_features)  # 加个 L1 loss 需要稀疏
        self.num_cotpye = num_cotpye
        self.num_node_features = num_node_features
        self.depth = depth
        self.nhead = nhead
        self.num_category = num_category
        self.num_negative = 1

        self.embedding = resnet18(pretrained=True, embedding_size=self.num_node_features)

        self.disentangle_gablock = CORL(attr_aspect=num_branch, node_dim=num_node_features, gnn_deep=depth,
                                        attention_head=nhead, attention_layer=num_layers,
                                        attention_feed_dim=dim_feedforward)

    def forward(self, data):

        # data.x: (N, 3, img_len, img_width) N为一个batch（包括正样本和负样本）的img_item数

        # get img embed
        # print(f"原data.x.shape: {data.x.shape}")
        data.x = self.embedding(data.x)  # (N, D), D 为 embedding

        # print(f"embedding后的data.x.shape: {data.x.shape}")
        img_embed_norm = data.x.norm(2) / np.sqrt(data.x.size(0))
        disentangle_data = data.clone()

        # attr branch
        fine_feature, fine_score, attr_type_mask_norm, diversity_loss = self.disentangle_gablock(disentangle_data)
        # fine_feature: (Graph_N,2D), fine_score: (Graph_N, 1), attr_type_mask_norm: 标量，为损失,
        # img_embed_norm为标准化后的embedding, diversity_loss分类损失
        # print(f"fine_score: {fine_score}")
        return fine_score, attr_type_mask_norm, img_embed_norm, diversity_loss

    def bpr_loss(self, output):
        # output.shape: (Graph_N, 1)
        # ****Graph_N = batch_size * (num_negative + 1)****
        # print(f"output: {output.shape}")
        output = output.view(-1, (self.num_negative + 1))  # each row: (pos, neg, neg, neg, ..., neg)

        # the first score (pos scores) minus each remainder scores (neg scores)
        # 用于计算贝叶斯个性化排名损失，原理：促进正类的评分比负类高
        output = output[:, 0].unsqueeze(-1).expand_as(output[:, 1:]) - output[:, 1:]

        # 大于 0 说明模型得到的正类评分比负类评分高，而这正是我们想要的结果（因为我们想要正类），所以用来算正确率
        batch_acc = (output > 0).sum().item() * 1.0 / output.nelement()

        return -F.logsigmoid(output).mean(), batch_acc

    @torch.no_grad()
    def test_fitb(self, batch):
        self.eval()
        output, _, _, _ = self(batch)
        output = output.view(-1, 4)  # each row: (pos, neg, neg, neg)
        _, max_idx = output.max(dim=-1)
        return (max_idx == 0).sum().item()

    @torch.no_grad()
    def test_auc(self, batch):
        self.eval()
        output, _, _, _ = self(batch)
        return output.view(-1)

    @torch.no_grad()
    def test_retrieval(self, batch, ranking_neg_num):
        self.eval()
        output = self(batch)[0]
        return output.view(-1, ranking_neg_num + 1)
