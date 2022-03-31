import torch
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from config import args_info, device
from models import CF
from utils import DataLoader, BenchmarkDataset
from utils.tools import Timer, model_save, get_lr, model_size, code_snapshot


def train(model, optimizer, train_loader, epoch, tb):
    model.train()
    for i, train_batch in enumerate(train_loader):
        train_batch = train_batch.to(device)

        # print(f"train_batch: {len(train_batch)}")
        # print(f"train_batch: {train_batch}")

        # ret = fine_score
        ret, type_mask_norm, img_embed_norm, diversity_norm = model(train_batch)

        # print(f'fine score: {ret.shape}')

        bpr_loss, batch_acc = model.bpr_loss(ret)

        loss_type_mask = 5e-4 * type_mask_norm
        loss_img_embed = 5e-3 * img_embed_norm
        loss_diversity = 5e-3 * diversity_norm
        loss = bpr_loss + loss_type_mask + loss_img_embed + loss_diversity

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not (i % 20):
            print(f"Epoch[{epoch + 1}][{i + 1}/{len(train_loader)}] "
                  f"loss: {loss.item():.5f} "
                  f"bpr: {bpr_loss.item():.5f} "
                  f"mask_norm: {loss_type_mask.item():.5f} "
                  f"embed_norm: {loss_img_embed.item():.5f} "
                  f"diversity_norm: {loss_diversity.item():.5f} "
                  f"acc: {batch_acc:.2f} ")

            iterations = epoch * len(train_loader) + i

            tb.add_scalar('data/losses/bpr', bpr_loss.item(), iterations)
            tb.add_scalar('data/losses/mask_norm', loss_type_mask.item(), iterations)
            tb.add_scalar('data/losses/embed_norm', loss_img_embed.item(), iterations)
            tb.add_scalar('data/losses/diversity_norm', loss_diversity.item(), iterations)

            tb.add_scalar('data/tot_loss', loss.item(), iterations)
            tb.add_scalar('data/train_acc', batch_acc, iterations)
            tb.add_scalar('data/lr', get_lr(optimizer), iterations)


def inference(model, auc_loader, fitb_loader, epoch, tb):
    model.eval()
    with torch.no_grad():
        scores, labels = [], []
        for auc_batch in auc_loader:
            auc_batch = auc_batch.to(device)
            auc_score = model.test_auc(auc_batch)
            scores.append(auc_score.cpu())
            labels.append(auc_batch.y.cpu())

        scores = torch.cat(scores).numpy()
        labels = torch.cat(labels).numpy()
        cp_auc = roc_auc_score(labels, scores)

        fitb_right = 0
        for fitb_batch in fitb_loader:
            fitb_batch = fitb_batch.to(device)
            fitb_right += model.test_fitb(fitb_batch)
        fitb_acc = fitb_right / len(fitb_loader.dataset)

    tb.add_scalars('data/kpi', {
        'auc': cp_auc,
        'fitb': fitb_acc
    }, epoch)

    total = cp_auc + fitb_acc

    return cp_auc, fitb_acc, total


def main(args):
    # tensorboard logging
    # web command: tensorboard --port=8097 --logdir tb_logs/exp1/ --bind_all
    tb = SummaryWriter(f'tb_logs/{args.remark}')

    # data preparation
    BenchmarkDataset.init(args)
    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': args.num_worker,
                     'pin_memory': True if torch.cuda.is_available() else False}
    train_dataset = BenchmarkDataset('train').next_train_epoch()
    auc_dataset_valid = BenchmarkDataset('valid').test_auc()
    fitb_dataset_valid = BenchmarkDataset('valid').test_fitb()
    auc_loader_valid = DataLoader(auc_dataset_valid, **loader_kwargs)
    fitb_loader_valid = DataLoader(fitb_dataset_valid, **loader_kwargs)

    # define model
    model = CF(num_node_features=args.hid, num_cotpye=train_dataset.num_cotpye,
               depth=args.gd, nhead=args.nhead, dim_feedforward=args.fdim,
               num_layers=arg.nlayer, num_category=train_dataset.num_semantic_category).to(device)

    # define optimize and adjust lr
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    ajlr = ExponentialLR(optimizer, gamma=1 - 0.015)

    # model size
    print(f'  + Size of params: {model_size(model):.2f}MB')

    # record best result
    best_auc, best_fitb, best_total = 0., 0., 0.
    for epoch in range(args.epoch):
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        Timer.start('epoch_turn')

        # train for one epoch
        train(model, optimizer, train_loader, epoch, tb)
        # evaluate on validation set
        cp_auc, fitb_acc, kpi_total = inference(model, auc_loader_valid, fitb_loader_valid, epoch, tb)
        # update learning rate
        ajlr.step()

        # remember best acc and save checkpoint
        is_best = kpi_total > best_total
        best_auc = max(best_auc, cp_auc)
        best_fitb = max(best_fitb, fitb_acc)
        best_total = max(best_total, kpi_total)

        best_path = model_save(args.remark, model, epoch, is_best, best_auc=cp_auc, best_fitb=fitb_acc)

        print(
            f"Epoch[{epoch + 1}][{Timer.end_time('epoch_turn'):.2f}s] "
            f"auc: {cp_auc:.4f} fitb: {fitb_acc:.4f} "
            f"best_auc: {best_auc:.4f} best_fitb: {best_fitb:.4f} ")

        # generate new negative training samples
        print('next epoch')
        train_dataset.next_train_epoch()

    # inference on test
    print('Train End!')
    auc_loader_test = DataLoader(BenchmarkDataset('test').test_auc(), **loader_kwargs)
    fitb_loader_test = DataLoader(BenchmarkDataset('test').test_fitb(), **loader_kwargs)

    ckpt = torch.load(best_path, map_location=torch.device('cuda'))
    model.load_state_dict(ckpt["state_dict"])
    cp_auc, fitb_acc, _ = inference(model, auc_loader_test, fitb_loader_test, epoch, tb)
    print(f"auc: {cp_auc:.4f} fitb: {fitb_acc:.4f} ")

    tb.close()


if __name__ == '__main__':
    arg = args_info()
    code_snapshot(arg.remark)
    main(arg)
