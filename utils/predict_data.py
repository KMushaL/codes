import os
import os.path as osp
import json
import pickle

from PIL import Image
from torch_geometric.data import Data
from torchvision import transforms
import torch
import numpy as np

from utils.dataloader import Batch


def image_loader(path):
    return Image.open(path).convert('RGB')


def load_typespaces(rootdir):
    """ loads a mapping of pairs of types to the embedding used to
        compare them
    """
    # typespaces保存了配对表
    # 如 ('bags', 'shoes'), ('bags', 'jewellery'), ...
    # data/polyvore_outfits/nondisjoint/typespaces.p
    typespace_fn = osp.join(rootdir, 'typespaces.p')
    with open(typespace_fn, 'rb') as fp:
        typespaces = pickle.load(fp)

    # ts: dict
    # {('bags', 'shoes'): 0, ('bags', 'jewellery'): 1, ...}
    ts = {}
    for index, t in enumerate(typespaces):
        ts[t] = index

    return ts


class PredidctData:
    # class vars
    _args = None
    _pred_root_dir = None
    _rec_dir = None  # 备选的推荐服装文件夹，默认为recommend_images
    _root_dir = None
    _image_dir = None
    _seman2dense = None
    _cate2dense = None
    _meta_data = None
    _class_init_flag = False
    _max_outfit_len = -1
    _call_next_epoch = 0

    __img_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
    _inference_transform = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        __img_normalize,
    ])

    @classmethod
    def init(cls, args, cloth_img_dir, cloth_index, recommend_img_dir=''):
        cls._class_init_flag = True
        cls._args = args

        # _args.data_dir = data
        # _root_dir = data/polyvore_outfits/nondisjoint
        cls._root_dir = osp.join(
            cls._args.data_dir, 'polyvore_outfits', cls._args.polyvore_split)

        cls._pred_root_dir = 'pred_data'

        # _image_dir = pred_data/images
        # cloth_index = one, two, three...
        cls._image_dir = osp.join(cls._pred_root_dir, cloth_img_dir, cloth_index)

        if recommend_img_dir != '':
            cls._rec_dir = osp.join(cls._pred_root_dir, recommend_img_dir)

        # _meta_data = data/polyvore_outfits/polyvore_item_metadata.json
        with open(osp.join(cls._args.data_dir, 'polyvore_outfits', 'polyvore_item_metadata.json'), 'r') as fp:
            cls._meta_data = json.load(fp)

        # preprocessing = data/polyvore_outfits/preprocessing.pkl
        with open(osp.join(cls._args.data_dir, 'polyvore_outfits', 'preprocessing.pkl'), 'rb') as fp:
            preprocessing = pickle.load(fp)

        # _cate2dense 和 _seman2dense 按照在 metedata.json 中类别出现的顺序重新给类别 id 进行赋值
        # 如第一件 item 的 category_id 为15，通过 _cate2dense 的映射后即变为 0
        cls._cate2dense = preprocessing.get("cate2dense")
        cls._seman2dense = preprocessing.get("seman2dense")

        # 使用 semantic_category 构建图
        cls._co_type_weight = cls._calc_co_weight(
            cls._seman2dense, 'semantic_category')

        # 构建item关系的图，若item_i和item_j共同出现则二者边权值+1，最后还要除以权值总和（归一化）

    @classmethod
    def _calc_co_weight(cls, dense_mapping, dense_key):
        r"""
        the weight of the static graph by data-driven manner.
        dense_mapping: 映射，如 semantic_dense 表示原 semantic 类别所人为控制而得到的映射的 id
        """
        num_category = len(dense_mapping)

        # 类别与类别之间构建图
        total_graph = np.zeros((num_category, num_category), dtype=np.float32)

        # outfit_data = data/polyvore_outfits/nondisjoint/train.json
        data_json = osp.join(cls._root_dir, 'train.json')
        with open(data_json, 'r') as fp:
            outfit_data = json.load(fp)

        # count co-concurrence times
        for outfit in outfit_data:
            cate_list = outfit['items']  # cate_list 包含了一套服装所出现的所有 item id
            cls._max_outfit_len = max(cls._max_outfit_len, len(cate_list))
            for i in range(len(cate_list)):
                # 获取 seman_category 对应的编号（即通过 _seman2dense 映射得到的编号，也对应 total_graph 的行与列）
                rcid = cls._get_im_dense_type(
                    cate_list[i]["item_id"], dense_mapping, dense_key)
                for j in range(i + 1, len(cate_list)):
                    rcjd = cls._get_im_dense_type(
                        cate_list[j]["item_id"], dense_mapping, dense_key)
                    # train.json 中的 "items" 中记录了同时出现的 item
                    total_graph[rcid][rcjd] += 1.
                    total_graph[rcjd][rcid] += 1.

        total_graph /= total_graph.sum(0)
        total_graph /= total_graph.sum(1, keepdims=True)

        return total_graph

    @classmethod
    def _get_im_dense_type(cls, im, which_dense, dense_key):
        r"""
        :param im: imamge id
        :param which_dense: 如 seman2dense ，其保存了原始的semantic类别对人为设置的 id 的映射
        :param dense_key: category_id 或 semantic_category
        :return: 返回人为设置的 category_id 或 semantic_id
        """
        # 因为 item 的类别信息保存在 _meta_data 中
        cate = cls._meta_data[im][dense_key]
        # 拿到原始的 category_id 或 semantic_category，再返回到人为设置的 id
        return which_dense[cate]

    def _get_typespace(self, anchor, pair):
        """ Returns the index of the type specific embedding
            for the pair of item types provided as input
            (anchor, pair) = (sema_raw_j, sema_raw_i)，原始的 semantic 类别（'top',...)
            j 连 i
            :return: self.typespaces 中 pair 所对应的 index
        """
        # self.typespaces = {('bags', 'shoes'): 0, ('bags', 'jewellery'): 1, ...}
        query = (anchor, pair)
        if query not in self.typespaces:
            query = (pair, anchor)  # 若不存在，则交换一下就可以

        return self.typespaces[query]

    # 构造GNN所需的Data类实例
    def _wrapper(self, file_items):
        """
        file_items: 一套服装的所有 item_id
        compatibility: True or False
        """
        index_source, index_target, edge_weight, type_embedding = [], [], [], []
        rcid_index = []
        scid_index = []
        for j, j_item in enumerate(file_items):
            # 获取第 j 件 item 对应的原始的 seman_category 和 cate_category id
            sema_raw_j, cate_raw_j = self.im2type[j_item]
            # 获取第 j 件 item 对应的经过重新编排后的 seman_category 和 cate_category id
            s_j_rcid, j_rcid = self._seman2dense[sema_raw_j], self._cate2dense[cate_raw_j]

            # 保存 file_items 中所有 item 的两个分类的 id
            rcid_index.append(j_rcid)
            scid_index.append(s_j_rcid)

            # i_item 为 j_item 所属服装的其他（除j_item之外）item
            for i, i_item in enumerate(file_items):
                if i == j:
                    continue

                # j 与 i 相连，j 为 source，i 为 target
                index_source.append(j)
                index_target.append(i)

                # 同上
                sema_raw_i, cate_raw_i = self.im2type[i_item]
                s_i_rcid, i_rcid = self._seman2dense[sema_raw_i], self._cate2dense[cate_raw_i]

                # 获取通过传入 seman_category 的 _calc_co_weight 方法而构建的图的边权值
                # probs of j, given i
                edge_weight.append(self._co_type_weight[s_j_rcid][s_i_rcid])
                # 如 ('bags', 'shoes'): 0
                # sema_raw_j = 'bags', sema_raw_i = 'shoes', 则 _get_typespace 返回 0
                type_embedding.append(
                    self._get_typespace(sema_raw_j, sema_raw_i))

        # print(torch.tensor(edge_weight).shape)
        # edge_index: 2x_ tensor，第一行为 index_source（边的起始点），第二行为 index_target（边的终点）
        data = Data(rcid_index=torch.tensor(rcid_index, dtype=torch.long),
                    scid_index=torch.tensor(scid_index, dtype=torch.long),
                    file_items=file_items,
                    edge_index=torch.tensor(
                        [index_source, index_target], dtype=torch.long),
                    edge_weight=torch.tensor(edge_weight, dtype=torch.float32),
                    type_embedding=torch.tensor(
                        type_embedding, dtype=torch.long))
        return data

    def _fetch_img(self, img_fns, _image_dir):
        """
        加载图片并对图片处理
        """
        apply_transform = self._inference_transform

        img_data = []
        for fn in img_fns:
            img = image_loader(osp.join(_image_dir, fn))
            img_data.append(apply_transform(img))

        # print(torch.stack(img_data, dim=0).shape)
        return torch.stack(img_data, dim=0)  # N,3,112,112，N为img_fns中图片的数量

    # 返回处理后的Data实例(一套服装)，用于预测分值
    def get_one_pred_data(self):
        images = os.listdir(self._image_dir)

        file_items = images.copy()
        file_items = [item.split('.')[0] for item in file_items]
        row_data = self._wrapper(file_items)

        row_data.x = self._fetch_img(images, self._image_dir)  # n, 3, 112, 112, n为一套服装的file_items的数量

        data_list = [[row_data]]
        data = Batch.from_data_list(data_list)

        one_cloth_path = [osp.join(self._image_dir, image) for image in images]  # 该套服装的全路径

        return one_cloth_path, data

    # 返回处理后的Data实例(多套服装)，用于预测分值
    def get_multi_pred_data(self):
        check_images = os.listdir(self._image_dir)  # 待推荐的服装(含后缀名)
        rec_images = os.listdir(self._rec_dir)  # 备选服装(含后缀名)

        check_items = check_images.copy()
        rec_items = rec_images.copy()

        check_items = [item.split('.')[0] for item in check_items]
        rec_items = [item.split('.')[0] for item in rec_items]

        check_images_fetch = self._fetch_img(check_images, self._image_dir)
        rec_images_fetch = self._fetch_img(rec_images, self._rec_dir)

        sum_file_items = []
        sum_images_fetch = []
        # sum_file_items = [[check_items_1,...,check_items_n, rec_items_1],
        #                   [check_items_1,...,check_items_n, rec_items_2],...]
        for i in range(len(rec_items)):
            sum_file_items.append(check_items.copy())
            sum_file_items[i].append(rec_items[i])
            sum_images_fetch.append(check_images_fetch)
            sum_images_fetch[i] = torch.cat((sum_images_fetch[i], rec_images_fetch[i].unsqueeze(dim=0)), dim=0)

        sum_row_data = [self._wrapper(file_item) for file_item in sum_file_items]

        data_list = []
        # data_list = [[Data()], [Data()], ..., [Data()]]
        for i in range(len(sum_row_data)):
            sum_row_data[i].x = sum_images_fetch[i]
            data_list.append([sum_row_data[i]])

        data = Batch.from_data_list(data_list)

        cloth_path = [osp.join(self._image_dir, image) for image in check_images]  # 待推荐的服装的全路径
        rec_images_path = [osp.join(self._rec_dir, rec_image) for rec_image in rec_images]  # 所有推荐服装的全路径
        return cloth_path, rec_images_path, data

    def __init__(self, split='train'):
        # spilt包括train, valid, test
        data_json = osp.join(self._root_dir, '%s.json' % split)
        with open(data_json, 'r') as fp:
            outfit_data = json.load(fp)

        # get list of images and make a mapping used to quickly organize the data
        im2type = {}

        for outfit in outfit_data:
            for item in outfit['items']:
                im = item['item_id']
                # _meta_data key值为 item_id
                seman_category = self._meta_data[im]['semantic_category']
                cate_category = self._meta_data[im]['category_id']
                # item_id 到 （seman_category 和 cate_category）的转换
                im2type[im] = (seman_category, cate_category)

        self.im2type = im2type  # item_id 到 （seman_category 和 cate_category）的转换
        # self.typespaces = {('bags', 'shoes'): 0, ('bags', 'jewellery'): 1, ...}
        self.typespaces = load_typespaces(self._root_dir)
