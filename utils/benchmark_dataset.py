# -*- coding: utf-8 -*-
# Date: 2020/10/20 14:13

"""
dataset from: https://github.com/mvasil/fashion-compatibility
"""
__author__ = 'tianyu'

import json
import os.path as osp
import pickle
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torchvision import transforms
from tqdm import tqdm

from utils.tools import Timer


def image_loader(path):
    return Image.open(path).convert('RGB')


def load_compatibility_questions(fn, id2im):
    """ Returns the list of compatibility questions for the split
        [([items],label)]

        fn: 1 210750761_1 210750761_2 210750761_3，1为label，后面的为 set_id _ item_index

        @return: [([items],label)], 如(['154249722', '188425631', '183214727'], 1)
        """
    with open(fn, 'r') as f:
        lines = f.readlines()

    compatibility_questions = []
    for line in lines:
        data = line.strip().split()
        compat_question = [id2im[iid] for iid in data[1:]]  # id2im 将 set_id _ item_index 转为对应的 item_id
        compatibility_questions.append((compat_question, int(data[0])))

    return compatibility_questions


def load_fitb_questions(fn, id2im):
    """ Returns the list of fill in the blank questions for the split
         [[P,N,N,N], [P,N,N,N] ]
         P:(items,label)
         N:(items,label)
         """
    # fn实例:
    # {"question": ["210750761_1", "210750761_2"], "blank_position": 3,
    #  "answers": ["210750761_3", "221049803_5", "218261368_6", "223682912_7"]}
    with open(fn, 'r') as f:
        data = json.load(f)
    questions = []
    for item in data:
        question = item['question']
        question_items = [id2im[iid] for iid in question]
        # 如上例子，right_id = 210750761_3
        right_id = f"{question[0].rsplit('_', maxsplit=1)[0]}_{item['blank_position']}"

        # id2im 将 set_id _ item_index 转为对应的 item_id
        # (['154249722', '188425631', '183214727'], True)
        PNNN = [(question_items.copy() + [id2im[right_id]], True)]
        answer = item['answers']
        for ans in answer:
            if ans == right_id:
                continue
            # 如上面将除 210750761_3 之外的答案作为 fill，并给予 label False
            # (['154249722', '188425631', '205505233'], False)
            PNNN.append((question_items.copy() + [id2im[ans]], False))

        questions.append(PNNN)

    return questions


def load_retrieval_questions(fn, id2im):
    """ Returns the list of fill in the blank questions for the split
         [[P,N,N,N], [P,N,N,N] ]
         P:(items,label)
         N:(items,label)
         与答案的item所属类别的其余item均为错误答案
         """
    with open(fn, 'r') as f:
        data = json.load(f)
    questions = []
    print('extract from disk ...')
    for item in tqdm(data):
        question = item['question']
        question_items = [id2im[iid] for iid in question]

        PNNN = [(question_items.copy() + [id2im[item['right']]], True)]
        answer = item['candidate']
        for ans in answer:
            # 与答案的item所属类别的其余item均为错误答案
            PNNN.append((question_items.copy() + [id2im[ans]], False))

        questions.append(PNNN)

    return questions


def load_typespaces(rootdir):
    """ loads a mapping of pairs of types to the embedding used to
        compare them
    """
    # typespaces保存了配对表
    # 如 ('bags', 'shoes'), ('bags', 'jewellery'), ...
    typespace_fn = osp.join(rootdir, 'typespaces.p')
    with open(typespace_fn, 'rb') as fp:
        typespaces = pickle.load(fp)

    # ts: dict
    # {('bags', 'shoes'): 0, ('bags', 'jewellery'): 1, ...}
    ts = {}
    for index, t in enumerate(typespaces):
        ts[t] = index

    return ts


class BenchmarkDataset(Dataset):
    # class vars
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
    _train_transform = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        __img_normalize,
    ])
    _inference_transform = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        __img_normalize,
    ])

    @classmethod
    def init(cls, args):
        cls._class_init_flag = True
        cls._args = args

        # _args.data_dir = data
        # _root_dir = data/polyvore_outfits/nondisjoint（或disjoint）
        cls._root_dir = osp.join(cls._args.data_dir, 'polyvore_outfits', cls._args.polyvore_split)

        # _image_dir = data/polyvore_outfits/images
        cls._image_dir = osp.join(cls._args.data_dir, 'polyvore_outfits', 'images')

        # _meta_data = data/polyvore_outfits/polyvore_item_metadata.json
        # _meta_data保存了item具体的信息：包括name, description, category...
        with open(osp.join(cls._args.data_dir, 'polyvore_outfits', 'polyvore_item_metadata.json'), 'r') as fp:
            cls._meta_data = json.load(fp)

        with open(osp.join(cls._args.data_dir, 'polyvore_outfits', 'preprocessing.pkl'), 'rb') as fp:
            preprocessing = pickle.load(fp)
        # _cate2dense 和 _seman2dense 按照在 metedata.json 中类别出现的顺序重新给类别 id 进行赋值
        # 如第一件 item 的 category_id 为15，通过 _cate2dense 的映射后即变为 0
        cls._cate2dense = preprocessing.get("cate2dense")
        cls._seman2dense = preprocessing.get("seman2dense")

        # 使用 semantic_category 构建图
        cls._co_type_weight = cls._calc_co_weight(cls._seman2dense, 'semantic_category')

    @property
    def num_category(self):
        return len(self._cate2dense)

    @property
    def num_semantic_category(self):
        return len(self._seman2dense)

    @property
    def num_cotpye(self):
        return len(self.typespaces)

    @classmethod
    def _get_im_dense_type(cls, im, which_dense, dense_key):
        r"""
        :param im: imamge id
        :param which_dense: 如 seman2dense ，其保存了原始的semantic类别对人为设置的 id 的映射
        :param dense_key: 如 category_id 或 semantic_category
        :return: 返回人为设置的 category_id 或 semantic_id
        """
        # _meta_data = data/polyvore_outfits/polyvore_item_metadata.json
        # 因为 item 的类别信息保存在 _meta_data 中
        cate = cls._meta_data[im][dense_key]
        # 拿到原始的 category_id 或 semantic_category，再返回到人为设置的 id
        return which_dense[cate]

    # 构建item关系的图，若item_i和item_j共同出现则二者边权值+1，最后还要除以权值总和（归一化）
    @classmethod
    def _calc_co_weight(cls, dense_mapping, dense_key):
        r"""
        the weight of the static graph by data-driven manner.
        dense_mapping: 映射，如 semantic_dense 表示原 semantic 类别所人为控制而得到的映射的 id
        """
        # data_json = data/polyvore_outfits/nondisjoint/train.json
        data_json = osp.join(cls._root_dir, 'train.json')
        with open(data_json, 'r') as fp:
            outfit_data = json.load(fp)

        num_category = len(dense_mapping)
        # 类别与类别之间构建图
        total_graph = np.zeros((num_category, num_category), dtype=np.float32)

        # count co-concurrence times
        # outfit_data = data/polyvore_outfits/nondisjoint/train.json
        for outfit in outfit_data:
            cate_list = outfit['items']  # cate_list 包含了一套服装所出现的所有 item id
            cls._max_outfit_len = max(cls._max_outfit_len, len(cate_list))
            for i in range(len(cate_list)):
                # 获取 seman_category 对应的编号（即通过 _seman2dense 映射得到的编号，也对应 total_graph 的行与列）
                rcid = cls._get_im_dense_type(cate_list[i]["item_id"], dense_mapping, dense_key)
                for j in range(i + 1, len(cate_list)):
                    rcjd = cls._get_im_dense_type(cate_list[j]["item_id"], dense_mapping, dense_key)
                    # train.json 中的 "items" 中记录了同时出现的 item
                    total_graph[rcid][rcjd] += 1.
                    total_graph[rcjd][rcid] += 1.

        total_graph /= total_graph.sum(0)
        total_graph /= total_graph.sum(1, keepdims=True)

        return total_graph

    def __init__(self, split):
        assert self._class_init_flag, f"Init:{self._class_init_flag}-> " \
                                      f"you must init class firstly by calling BenchmarkDataset.init(args)"

        # _root_dir = data/polyvore_outfits/nondisjoint/split.json
        # spilt包括train, valid和test
        data_json = osp.join(self._root_dir, '%s.json' % split)
        # outfit_data = data/polyvore_outfits/nondisjoint/split
        with open(data_json, 'r') as fp:
            outfit_data = json.load(fp)

        # get list of images and make a mapping used to quickly organize the data
        im2type = {}
        category2ims = {}
        # imnames 保存 所有（服装）的所有 item_id
        imnames = set()
        id2im = {}
        for outfit in outfit_data:
            outfit_id = outfit['set_id']  # 一套服装的id
            for item in outfit['items']:
                im = item['item_id']
                # _meta_data = data/polyvore_outfits/polyvore_item_metadata.json
                # _meta_data保存了item具体的信息：包括name, description, category...
                # _meta_data key值为 item_id
                seman_category = self._meta_data[im]['semantic_category']
                cate_category = self._meta_data[im]['category_id']
                # item_id 到 （seman_category 和 cate_category）的转换
                im2type[im] = (seman_category, cate_category)

                # seman_category 到 item_id 的转换（不同的 item_id 的 seman_category 可能相同）
                category2ims.setdefault(seman_category, [])
                category2ims[seman_category].append(im)

                # (一套服装的) set_id _ item_id的索引（从1开始，依次递增）到 item_id 的转换
                # 如set_id为123456，item_id为555，其为该套服装的第一件item，即index为1，则id2im['123456_1']=555
                id2im['%s_%i' % (outfit_id, item['index'])] = im

                # imnames 保存 所有（服装）的所有 item_id
                imnames.add(im)

        imnames = list(imnames)
        self.imnames = imnames  # imnames 保存 所有（服装）的所有 item_id
        self.im2type = im2type  # item_id 到 （seman_category 和 cate_category）的转换
        self.id2im = id2im  # (一套服装的) set_id _ item_id的索引（从1开始，依次递增）到 item_id 的转换
        self.category2ims = category2ims  # seman_category 到 item_id 的转换（不同的 item_id 的 seman_category 可能相同）
        self.split = split
        # self.typespaces = {('bags', 'shoes'): 0, ('bags', 'jewellery'): 1, ...}
        self.typespaces = load_typespaces(self._root_dir)

        self.num_negative = None
        if self.split == 'train':
            # pos_list 为采集的正样本
            self.pos_list = self._collect_pos_sample(outfit_data)
            self.neg_list = None
        else:
            self.kpi_list = None  # kpi_list 保存了所有 fit_blank 或 retrieval 中所有答案样本（包括正负）所构建的图数据

    def next_train_epoch(self, num_negative=1):
        self._call_next_epoch += 1

        begin_same, begin_one = 10, 40
        if self._call_next_epoch < begin_same:
            print('+++ rand sample ++')
        elif self._call_next_epoch < begin_one:
            print('+++ same type sample ++')
        else:
            print('+++ replace one sample ++')
        self.num_negative = num_negative
        Timer.start('train_neg')
        # negative sample strategy: random -> same type -> replace
        self.neg_list = self._generate_neg_sample_rand2same2one(begin_same, begin_one)
        Timer.end('train_neg')

        return self

    def test_auc(self, file_name_format="compatibility_%s.txt"):
        ret = []
        fn = osp.join(self._root_dir, file_name_format % self.split)
        compatibility_questions = load_compatibility_questions(fn, self.id2im)

        for items, label in compatibility_questions:
            ret_data = self._wrapper(items, label)
            ret.append([ret_data])

        self.kpi_list = ret
        return self

    def test_fitb(self, file_name_format='fill_in_blank_%s.json'):
        ret = []
        fn = osp.join(self._root_dir, file_name_format % self.split)
        fitb_questions = load_fitb_questions(fn, self.id2im)
        for fitb in fitb_questions:
            # fitb: [P,N,N,N]，一个正确答案，三个错误答案
            # 如fitb:
            # [(['154249722', '188425631', '183214727'], True), (['154249722', '188425631', '205505233'], False),
            #  (['154249722', '188425631', '191902752'], False), (['154249722', '188425631', '195679004'], False)]
            post_fitb = []
            # each: (['154249722', '188425631', '183214727'], True)
            for each in fitb:
                # each: (items, label)
                ret_data = self._wrapper(*each)  # *each = ['154249722', '188425631', '183214727'] True
                post_fitb.append(ret_data)
            # post_fitb: [P,N,N,N]
            ret.append(post_fitb)

        # kpi_list 保存了所有 fit_blank 答案样本所构建的图数据
        self.kpi_list = ret
        return self

    def test_retrieval(self):
        ret = []
        fn = osp.join(self._args.data_dir, 'polyvore_outfits', 'retrieval', "{}.json".format(self._args.polyvore_split))
        retrieval_questions = load_retrieval_questions(fn, self.id2im)

        print('pack to kpi list ...')
        for fitb in tqdm(retrieval_questions):
            # fitb: [P,N,N,N]
            post_fitb = []
            for each in fitb:
                # each: (items, label)
                ret_data = self._wrapper(*each)
                post_fitb.append(ret_data)
            # post_fitb: [P,N,N,N]
            ret.append(post_fitb)

        self.kpi_list = ret
        return self

    # 采集正样本
    # outfit_data = data/polyvore_outfits/nondisjoint/split
    # spilt包括train, valid和test
    def _collect_pos_sample(self, outfit_data):
        ret = []
        for outfit in outfit_data:
            # items保存了一套服装的所有item_id（一套服装的所有item_id都单独构成一个list），如：
            # ['188852454', '203345917']
            items = list(map(lambda dic: dic["item_id"], outfit['items']))
            ret_data = self._wrapper(items, True)
            ret.append(ret_data)

        return ret

    # 构造GNN所需的Data类实例
    def _wrapper(self, file_items, compatibility):
        """
        file_items: 一套服装的所有 item_id
        compatibility: True or False
        """
        index_source, index_target, edge_weight, type_embedding, y = [], [], [], [], [int(compatibility)]
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
                edge_weight.append(self._co_type_weight[s_j_rcid][s_i_rcid])  # probs of j, given i
                # 如 ('bags', 'shoes'): 0
                # sema_raw_j = 'bags', sema_raw_i = 'shoes', 则 _get_typespace 返回 0
                type_embedding.append(self._get_typespace(sema_raw_j, sema_raw_i))

        # edge_index: 2x2 tensor，第一行为 index_source（边的起始点），第二行为 index_target（边的终点）
        data = Data(rcid_index=torch.tensor(rcid_index, dtype=torch.long),
                    scid_index=torch.tensor(scid_index, dtype=torch.long),
                    file_items=file_items,
                    edge_index=torch.tensor([index_source, index_target], dtype=torch.long),
                    edge_weight=torch.tensor(edge_weight, dtype=torch.float32),
                    type_embedding=torch.tensor(type_embedding, dtype=torch.long),
                    y=torch.tensor(y).float())
        return data

    def _generate_neg_sample_rand2same2one(self, begin_same, begin_one):
        ret = []
        for i, pos in enumerate(self.pos_list):
            if self._call_next_epoch < begin_same:
                ret.append(self._neg_rand(i, len(pos["file_items"]), self.num_negative))
            elif self._call_next_epoch < begin_one:
                ret.append(self._neg_rand_same_type(i, pos["file_items"], self.num_negative))
            else:
                ret.append(self._neg_rand_same_type_one(i, pos["file_items"], self.num_negative))

        print('negative sample done!')
        return ret

    def _neg_rand(self, i, pos_len, num_negative):
        """
        pos_len: len(pos["file_items"])
        num_negative: 负样本数量
        """
        if i and i % 5000 == 0:
            print(f"neg sample at {i}")
        neg_outfits = []
        neg_i = 0

        tot_len = len(self.imnames)  # 所有 item_id 的数量
        while neg_i < num_negative:
            neg_len = pos_len
            neg_outfit_ids = []
            for i in range(neg_len):
                while True:
                    # 获取任意一个 item_id ，只要不在负样本中，就添加进去（随机采样）
                    nno = np.random.randint(0, tot_len)
                    neg = self.imnames[nno]
                    if neg not in neg_outfit_ids:  # no the same item in one outfit
                        break

                neg_outfit_ids.append(neg)
            # construct Data
            neg_data = self._wrapper(neg_outfit_ids, False)
            neg_outfits.append(neg_data)

            neg_i += 1
        return neg_outfits

    def _neg_rand_same_type(self, i, file_items, num_negative):
        """
        file_items: pos["file_items"]
        num_negative: 负样本数量
        return: 返回同类别的保存了 item_id 的负样本
        """
        if i and i % 5000 == 0:
            print(f"neg sample at {i}")
        neg_outfits = []
        neg_i = 0

        while neg_i < num_negative:
            neg_len = len(file_items)
            neg_outfit_ids = []
            for i in range(neg_len):
                while True:
                    sem, _ = self.im2type[file_items[i]]  # 获取 item_id 的 seman_category 类别
                    candidate_sets = self.category2ims[sem]  # 属于该类别的所有 item_id 的集合
                    nno = np.random.randint(0, len(candidate_sets))
                    neg = candidate_sets[nno]

                    if neg not in neg_outfit_ids:  # no the same item in one outfit
                        break

                neg_outfit_ids.append(neg)
            # construct Data
            neg_data = self._wrapper(neg_outfit_ids, False)
            neg_outfits.append(neg_data)

            neg_i += 1
        return neg_outfits

    def _neg_rand_same_type_one(self, i, file_items, num_negative):
        """
        file_items: pos["file_items"]
        num_negative: 负样本数量
        return: 将一套服装其中的 item 换成其对应类别除其之外的 item
        """
        def com_sample():
            """
            return: 随机获取需要替换的位置列表
            """
            tmp_neg_iid = []
            tmp_num_neg = num_negative
            if neg_len < num_negative:
                while neg_len < tmp_num_neg:
                    tmp_neg_iid += random.sample(range(neg_len), k=neg_len)  # 将 range(neg_len) 这个列表打乱并返回长度为 k 的其中部分列表
                    tmp_num_neg -= neg_len  # 重复 num_negative / neg_len 次

            return tmp_neg_iid + random.sample(range(neg_len), k=tmp_num_neg)  # 返回剩余的含 num_negative % neg_len 个数目的打乱的列表

        if i and i % 5000 == 0:
            print(f"neg sample at {i}")
        neg_outfits = []
        neg_i = 0

        neg_len = len(file_items)
        neg_replace_pos = com_sample()  # 需替换的位置index
        while neg_i < num_negative:
            neg_outfit_ids = file_items.copy()
            while True:
                sem, _ = self.im2type[file_items[neg_replace_pos[neg_i]]]
                candidate_sets = self.category2ims[sem]  # 属于该类别的所有 item_id 的集合
                nno = np.random.randint(0, len(candidate_sets))
                neg = candidate_sets[nno]

                if neg not in neg_outfit_ids:  # no the same item in one outfit
                    break

            neg_outfit_ids[neg_replace_pos[neg_i]] = neg  # 替换

            # construct Data
            neg_data = self._wrapper(neg_outfit_ids, False)
            neg_outfits.append(neg_data)

            neg_i += 1
        return neg_outfits

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

    def _fetch_img(self, img_fns):
        """
        加载图片并对图片处理
        """
        if self.split == 'train':
            apply_transform = self._train_transform
        else:
            apply_transform = self._inference_transform

        img_data = []
        for fn in img_fns:
            img = image_loader(osp.join(self._image_dir, fn))
            img_data.append(apply_transform(img))

        # print(torch.stack(img_data, dim=0).shape)
        return torch.stack(img_data, dim=0)  # N,3,112,112，N为img_fns中图片的数量

    def __getitem__(self, index):
        if self.split == 'train':
            bundle = [self.pos_list[index].clone()] + [obj.clone() for obj in self.neg_list[index]]
        else:
            bundle = [obj.clone() for obj in self.kpi_list[index]]
        # print(f"self.pos_list[index]: {self.pos_list[index]}")
        # print(f"self.neg_list[index]: {self.neg_list[index]}")
        # print(f"bundle: {bundle}")
        for one in bundle:
            # print(f"one: {one}")
            img_fns = [f"{iid}.jpg" for iid in one["file_items"]]
            one.x = self._fetch_img(img_fns)  # N, 3, 112, 112, N为img_fns中图片即file_items的数量
            # print(f"one.x: {one.x.shape}")
        return bundle

    def __len__(self):
        if self.split == 'train':
            return len(self.pos_list)

        return len(self.kpi_list)
