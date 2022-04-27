import random
from config import args_info, device
from utils.test_utils import *


def test_score_and_history(model, cloth_index='one'):
    one_cloth_path, one_cloth_data = load_one_cloth_data(args, 'images', cloth_index)
    one_cloth_data = one_cloth_data.to(device)
    # one_cloth_data: Batch(edge_index=[2, 6], edge_weight=[6, 1], rcid_index=[3], scid_index=[3],
    # slices_indicator=[2], type_embedding=[6], x=[3, 3, 112, 112])
    score = get_compatibility_score(model, one_cloth_data)
    real_score = round(compute_real_score(score).item(), 2)
    if real_score < 10:
        real_score = real_score * 10
    do_score_history(one_cloth_path, real_score)
    return real_score


def test_recommend_and_history(model, cloth_index='one'):
    cloth_path, rec_images_path, multi_cloth_data = load_multi_cloth_data(args, 'check_suit', cloth_index,
                                                                          'recommend_images')
    multi_cloth_data = multi_cloth_data.to(device)
    # multi_cloth_data: Batch(edge_index=[2, 18], edge_weight=[18, 1], rcid_index=[9], scid_index=[9],
    # slices_indicator=[4], type_embedding=[18], x=[9, 3, 112, 112])
    score = get_compatibility_score(model, multi_cloth_data)
    real_score = compute_real_score(score)
    rec_index = torch.argmax(real_score)  # 获取分值最大的服装索引，该索引即为推荐的服装
    rec_img_path = rec_images_path[rec_index]  # 获取推荐的服装图片完整路径
    do_recommend_history(cloth_path, rec_img_path)
    return rec_img_path


def main(args):
    cloth_index = random.choice(cloth_list)
    model = load_model(args)
    real_score = test_score_and_history(model, cloth_index)
    rec_cloth_img_path = test_recommend_and_history(model, cloth_index)
    print(f"real_score: {real_score}")
    print(f"rec_cloth_img_path: {rec_cloth_img_path}")


if __name__ == '__main__':
    args = args_info()
    main(args)
