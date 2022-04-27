import json
import os

import torch

from config import device
from models import CF
from utils import BenchmarkDataset
from utils.predict_data import PredidctData

cloth_list = ['one', 'two', 'three']

score_history_dir = 'score_history'
rec_history_dir = 'recommend_history'


def compute_real_score(score):
    """

    :param score: 模型输出的分数
    :return: 真实分数
    """
    return torch.abs(score) / 3 * 100


def do_score_history(cloth_paths, score):
    img_name = [cloth_path.split('\\')[-1] for cloth_path in cloth_paths]
    score_dict = {'items': img_name, 'score': score}
    # 写入json文件
    with open(f'{score_history_dir}/record_{len(os.listdir(score_history_dir)) + 1}.json', 'w') as f:
        json.dump(score_dict, f)


def do_recommend_history(cloth_paths, rec_path):
    img_name = [cloth_path.split('\\')[-1] for cloth_path in cloth_paths]
    rec_img_name = rec_path.split('\\')[-1]
    rec_dict = {'items': img_name, 'rec_item': rec_img_name}
    # 写入json文件
    with open(f'{rec_history_dir}/record_{len(os.listdir(rec_history_dir)) + 1}.json', 'w') as f:
        json.dump(rec_dict, f)


def load_one_cloth_data(args, cloth_img_dir, cloth_index):
    """
    获得一套待预测兼容性评分的服装，用于传输给模型
    :param args:
    :param cloth_img_dir: 服装图片根路径，默认为pred_data下的images文件夹
    :param cloth_index: 第几套（images文件夹下的one，two，three）服装
    :return: 返回一套服装数据
    """
    PredidctData.init(args, cloth_img_dir, cloth_index)
    cloth_path, pred_data = PredidctData().get_one_pred_data()
    return cloth_path, pred_data


def load_multi_cloth_data(args, cloth_img_dir, cloth_index, recommend_img_dir):
    """
    传输一套有缺失服装单品的一套服装，然后返回其与recommend_img_dir文件夹下所有服装单品构建成的一套服装的数据
    :param args:
    :param cloth_img_dir: 服装图片根路径，默认为pred_data下的check_suit文件夹
    :param cloth_index: 第几套（check_suit文件夹下的one，two，three）服装
    :param recommend_img_dir: 备选服装路径
    :return: 返回待推荐服装图片完整路径、备选图片完整路径和多套服装数据
    """
    PredidctData.init(args, cloth_img_dir, cloth_index, recommend_img_dir)
    cloth_path, rec_images_path, data = PredidctData().get_multi_pred_data()
    # rec_images保存了所有的备选图片完整路径
    return cloth_path, rec_images_path, data


def load_model(args):
    """
    加载模型
    :param args: config.py中设置的参数
    :return: 返回加载完训练最优参数的模型
    """
    # data preparation
    BenchmarkDataset.init(args)

    train_dataset = BenchmarkDataset('train').next_train_epoch()
    # define model
    model = CF(num_node_features=args.hid, num_cotpye=train_dataset.num_cotpye,
               depth=args.gd, nhead=args.nhead, dim_feedforward=args.fdim,
               num_layers=args.nlayer, num_category=train_dataset.num_semantic_category).to(device)

    # 加载训练完的最优参数
    ckpt = torch.load("checkpoints/exp_1648963426/best_model.pt", map_location=torch.device('cuda'))
    model.load_state_dict(ckpt["state_dict"])

    return model


def get_compatibility_score(model, cloth_data):
    """
    评估一套服装的兼容性分值
    :param model: 模型
    :param cloth_data: 待传入模型的属于一(多)套服装的服装单品数据，(N,3,img_len,img_width)
    :return: 兼容性评分
    """
    model.eval()
    score = model.test_auc(cloth_data).cpu()
    return score
