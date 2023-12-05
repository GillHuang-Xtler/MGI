import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
from utils.data_processing import set_idx, label_mapping
import numpy as np
from .game import NashMSFL
from utils.data_processing import total_variation
from torchmetrics.image import TotalVariation
from utils.save import save_img, early_stop, save_final_img, save_eval

from .method_utils import gradient_closure, gradient_closure2


def dlg(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path, str_time):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1
    net = nets[0]

    for imidx in range(num_dummy):
        idx, imidx_list = set_idx(imidx, imidx_list, idx_shuffle)
        tmp_datum = tt(dst[idx][0]).float().to(device)
        tmp_datum = tmp_datum.view(1, *tmp_datum.size())
        tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
        tmp_label = tmp_label.view(1, )
        if imidx == 0:
            gt_data = tmp_datum
            gt_label = tmp_label
        else:
            gt_data = torch.cat((gt_data, tmp_datum), dim=0)
            gt_label = torch.cat((gt_label, tmp_label), dim=0)

    out = net(gt_data)
    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    # print(dummy_data.size())
    dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr= args.lr)
    history = []
    history_iters = []
    # losses = []
    train_iters = []
    results = []
    args.logger.info('lr = #{}', args.lr)
    for iters in range(Iteration):
        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff
        current_loss = optimizer.step(closure)
        train_iters.append(iters)
        # losses.append(current_loss)

        result = save_eval(args.get_eval_metrics(), dummy_data, gt_data)
        if iters % 100 == 0:
            args.logger.info('iters idx: #{}, current lr: #{}', iters, optimizer.param_groups[0]['lr'])
            args.logger.info('loss: #{}, mse: #{}, lpips: #{}, psnr: #{}, ssim: #{}', current_loss, result[0], result[1], result[2], result[3])
        results.append(result)

        if args.earlystop > 0:
            _es = early_stop(args, current_loss, iters, tp, final_img, dummy_data, imidx_list, imidx, num_dummy, save_path)
            if _es:
                break

        # save the training image
        if iters % int(Iteration / args.log_interval) == 0:
            save_img(iters, args, history, tp, dummy_data, num_dummy, history_iters, gt_data, save_path, imidx_list, str_time, mean_std)

    # label = torch.argmax(dummy_label, dim=-1).detach().item()
    args.logger.info("inversion finished")

    return imidx_list, final_iter, final_img, results


def idlg(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path, str_time):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1
    net = nets[0]


    for imidx in range(num_dummy):
        idx, imidx_list = set_idx(imidx, imidx_list, idx_shuffle)
        tmp_datum = tt(dst[idx][0]).float().to(device)
        tmp_datum = tmp_datum.view(1, *tmp_datum.size())
        tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
        tmp_label = tmp_label.view(1, )
        if imidx == 0:
            gt_data = tmp_datum
            gt_label = tmp_label
        else:
            gt_data = torch.cat((gt_data, tmp_datum), dim=0)
            gt_label = torch.cat((gt_label, tmp_label), dim=0)

    out = net(gt_data)
    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    # do not reconstruct label
    # dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data, ], lr=args.lr)

    # predict the ground-truth label
    label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
        (1,)).requires_grad_(False)
    history = []
    history_iters = []
    # losses = []
    train_iters = []
    results = []
    args.logger.info('lr = #{}', args.lr)
    for iters in range(Iteration):
        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            dummy_loss = criterion(pred, label_pred)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff
        current_loss = optimizer.step(closure)
        train_iters.append(iters)
        # losses.append(current_loss)
        result = save_eval(args.get_eval_metrics(), dummy_data, gt_data)
        if iters % 100 == 0:
            args.logger.info('iters idx: #{}, current lr: #{}', iters, optimizer.param_groups[0]['lr'])
            args.logger.info('loss: #{}, mse: #{}, lpips: #{}, psnr: #{}, ssim: #{}', current_loss, result[0], result[1], result[2], result[3])
        results.append(result)

        if args.earlystop > 0:
            _es = early_stop(args, current_loss, iters, tp, final_img, dummy_data, imidx_list, imidx, num_dummy,
                             save_path)
            if _es:
                break

        # save the training image
        if iters % int(Iteration / args.log_interval) == 0:
            save_img(iters, args, history, tp, dummy_data, num_dummy, history_iters, gt_data, save_path, imidx_list,
                     str_time, mean_std)

    # label = torch.argmax(dummy_label, dim=-1).detach().item()
    args.logger.info("inversion finished")

    return imidx_list, final_iter, final_img, results


def dlgadam(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path, str_time):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1
    net = nets[0]

    for imidx in range(num_dummy):
        idx, imidx_list = set_idx(imidx, imidx_list, idx_shuffle)
        tmp_datum = tt(dst[idx][0]).float().to(device)
        tmp_datum = tmp_datum.view(1, *tmp_datum.size())
        tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
        tmp_label = tmp_label.view(1, )
        if imidx == 0:
            gt_data = tmp_datum
            gt_label = tmp_label
        else:
            gt_data = torch.cat((gt_data, tmp_datum), dim=0)
            gt_label = torch.cat((gt_label, tmp_label), dim=0)

    out = net(gt_data)
    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([dummy_data, dummy_label], lr= args.lr)
    history = []
    history_iters = []

    train_iters = []
    results = []
    for iters in range(Iteration):

        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff
        current_loss = optimizer.step(closure)

        train_iters.append(iters)

        result = save_eval(args.get_eval_metrics(), dummy_data, gt_data)

        if iters % 100 == 0:
            args.logger.info('iters idx: #{}, current lr: #{}', iters, optimizer.param_groups[0]['lr'])
            args.logger.info('loss: #{}, mse: #{}, lpips: #{}, psnr: #{}, ssim: #{}', current_loss, result[0], result[1], result[2], result[3])
        results.append(result)

        if args.earlystop > 0:
            _es = early_stop(args, current_loss, iters, tp, final_img, dummy_data, imidx_list, imidx, num_dummy,
                             save_path)
            if _es:
                break

        # save the training image
        if iters % int(Iteration / args.log_interval) == 0:
            save_img(iters, args, history, tp, dummy_data, num_dummy, history_iters, gt_data, save_path, imidx_list,
                     str_time, mean_std)

    args.logger.info("inversion finished")

    return imidx_list, final_iter, final_img, results


def invg(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path, str_time):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1
    net = nets[0]

    for imidx in range(num_dummy):
        idx, imidx_list = set_idx(imidx, imidx_list, idx_shuffle)

        tmp_datum = tt(dst[idx][0]).float().to(device)
        tmp_datum = tmp_datum.view(1, *tmp_datum.size())
        tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
        tmp_label = tmp_label.view(1, )
        if imidx == 0:
            gt_data = tmp_datum
            gt_label = tmp_label
        else:
            gt_data = torch.cat((gt_data, tmp_datum), dim=0)
            gt_label = torch.cat((gt_label, tmp_label), dim=0)

    out = net(gt_data)
    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([dummy_data, dummy_label], lr= args.lr)
    history = []
    history_iters = []
    # losses = []
    train_iters = []
    results = []
    args.logger.info('lr = #{}', args.lr)
    for iters in range(Iteration):

        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                cos = nn.CosineSimilarity(dim = 0)
                grad_diff += (1-cos(gx, gy)).sum()
            grad_diff += args.tv * total_variation(dummy_data)
            grad_diff.backward()
            return grad_diff
        current_loss = optimizer.step(closure)
        train_iters.append(iters)
        # losses.append(current_loss)
        result = save_eval(args.get_eval_metrics(), dummy_data, gt_data)
        if iters % 100 == 0:
            args.logger.info('iters idx: #{}, current lr: #{}', iters, optimizer.param_groups[0]['lr'])
            args.logger.info('loss: #{}, mse: #{}, lpips: #{}, psnr: #{}, ssim: #{}', current_loss, result[0], result[1], result[2], result[3])
        results.append(result)
        #
        if args.earlystop > 0:
            _es = early_stop(args, current_loss, iters, tp, final_img, dummy_data, imidx_list, imidx, num_dummy,
                             save_path)
            if _es:
                break

        # save the training image
        if iters % int(Iteration / args.log_interval) == 0:
            save_img(iters, args, history, tp, dummy_data, num_dummy, history_iters, gt_data, save_path, imidx_list,
                     str_time, mean_std)

    # label = torch.argmax(dummy_label, dim=-1).detach().item()
    args.logger.info("inversion finished")
    return imidx_list, final_iter, final_img, results


def mdlg(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path, str_time):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1

    d_mean, d_std = mean_std
    dm = torch.as_tensor(d_mean, dtype=next(nets[0].parameters()).dtype)[:, None, None].cuda()
    ds = torch.as_tensor(d_std, dtype=next(nets[0].parameters()).dtype)[:, None, None].cuda()

    for imidx in range(num_dummy):

        # get random idx or
        idx, imidx_list = set_idx(imidx, imidx_list, idx_shuffle)

        tmp_datum = tt(dst[idx][0]).float().to(device)
        tmp_datum = tmp_datum.view(1, *tmp_datum.size())
        tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
        tmp_label = tmp_label.view(1, )

        # use a random data as initialization
        # dummy_data = tt(dst[999][0]).float().to(device)
        # dummy_data = dummy_data.view(1, *dummy_data.size())

        if imidx == 0:
            gt_data = tmp_datum
            gt_label = tmp_label
        else:
            gt_data = torch.cat((gt_data, tmp_datum), dim=0)
            gt_label = torch.cat((gt_label, tmp_label), dim=0)

    # compute original gradient

    original_dy_dxs = []
    _label_preds = []

    for i in range(args.num_servers):
        out = nets[i](gt_data)
        y = criterion(out, gt_label)
        dy_dx = torch.autograd.grad(y, nets[i].parameters())
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        original_dy_dxs.append(original_dy_dx)

        # predict the ground-truth label
        _label_preds.append(torch.argmin(torch.sum(original_dy_dx[-2], dim=-1) , dim=-1).detach().reshape((1,)).requires_grad_(False))

    label_preds = []
    for i in _label_preds:
        j = i.repeat(args.num_dummy)
        label_preds.append(j)

    # initialize random image
    dummy_data = torch.randn(gt_data.size(), dtype=next(nets[0].parameters()).dtype).to(device).requires_grad_(True)
    # dummy_data = torch.rand(gt_data.size()).to(device).requires_grad_(True)
    # dummy_data = torch.randint(255, gt_data.size()).float().to(device)

    with torch.no_grad():
        dummy_data.data = torch.max(torch.min(dummy_data, (1 - dm) / ds), -dm / ds)

    dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
    if args.num_dummy > 1:
        if args.optim == 'LBFGS':
            optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr = args.lr)
        else:
            optimizer = optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=args.lr)
    else:
        if args.optim == 'LBFGS':
            optimizer = torch.optim.LBFGS([dummy_data, ], lr = args.lr)
        else:
            optimizer = optimizer = torch.optim.Adam([dummy_data, ], lr=args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=[args.iteration // 2.667, args.iteration // 1.6, args.iteration // 1.142], gamma=0.1)

    history = []
    history_iters = []
    train_iters = []
    results = []
    args.logger.info('lr = #{}', args.lr)
    for iters in range(Iteration):
        def closure():
            optimizer.zero_grad()
            # new
            dummy_dy_dxs = []
            for i in range(args.num_servers):
                pred = nets[i](dummy_data)
                if args.num_dummy > 1:
                    dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=1))
                else:
                    dummy_loss = criterion(pred, label_preds[i])
                dummy_dy_dxs.append(torch.autograd.grad(dummy_loss, nets[i].parameters(), create_graph=True))

            grad_diff = 0
            for i in range(args.num_servers):
                for gx, gy in zip(dummy_dy_dxs[i], original_dy_dxs[i]):
                    grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        if args.inv_loss == 'l2':
            current_loss = optimizer.step(closure)
        else:  # 'sim'
            current_loss = optimizer.step(gradient_closure2(optimizer, dummy_data, original_dy_dxs,
                                                        label_preds, nets, args, criterion))
        if args.optim == 'Adam':
            scheduler.step()

        with torch.no_grad():
            # Project into image space
            dummy_data.data = torch.max(torch.min(dummy_data, (1 - dm) / ds), -dm / ds)

            if (iters + 1 == args.iteration) or iters % 500 == 0:
                print(f'It: {iters}. Rec. loss: {current_loss.item():2.4f}.')

        train_iters.append(iters)
        result = save_eval(args.get_eval_metrics(), dummy_data, gt_data)
        if iters % 100 == 0:
            args.logger.info('iters idx: #{}, current lr: #{}', iters, optimizer.param_groups[0]['lr'])
            args.logger.info('loss: #{}, mse: #{}, lpips: #{}, psnr: #{}, ssim: #{}', current_loss, result[0], result[1], result[2], result[3])
        results.append(result)

        if args.earlystop > 0:
            _es = early_stop(args, current_loss, iters, tp, final_img, dummy_data, imidx_list, imidx, num_dummy,
                             save_path)
            if _es:
                break

        # save the training image
        if iters % int(Iteration / args.log_interval) == 0:
            save_img(iters, args, history, tp, dummy_data, num_dummy, history_iters, gt_data, save_path, imidx_list,
                     str_time, mean_std)

    args.logger.info("inversion finished")

    return imidx_list, final_iter, final_img, results

def mdlg_mt(args, device, num_dummy, idx_shuffle, tt, tp, dst, nets, num_classes, Iteration, save_path, str_time):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1
    tmp_labels = []
    gt_labels = []

    for imidx in range(num_dummy):
        idx, imidx_list = set_idx(imidx, imidx_list, idx_shuffle)
        tmp_datum = tt(dst[idx][0]).float().to(device)
        tmp_datum = tmp_datum.view(1, *tmp_datum.size())
        tmp_labels.append(torch.Tensor([dst[idx][1]]).long().to(device))

        '''get new mapping label for the same data sample'''
        if args.num_servers > 1:
            for i in range(args.num_servers-1):
                tmp_labels.append(label_mapping(origin_label = dst[idx][1], idx = i).to(device))
        # tmp_label_1 = torch.Tensor([dst[idx][1]]).long().to(device)

        for i in range(args.num_servers):
            tmp_labels[i] = tmp_labels[i].view(1, )

        if imidx == 0:
            gt_data = tmp_datum
            for i in range(args.num_servers):
                gt_labels.append(tmp_labels[i])
        else:
            gt_data = torch.cat((gt_data, tmp_datum), dim=0)
            for i in range(len(args.num_servers)):
                gt_labels.append(torch.cat((gt_labels[i], tmp_labels[i]), dim=0))
    # compute original gradient

    original_dy_dxs = []
    label_preds = []
    for i in range(args.num_servers):
        out = nets[i](gt_data)
        y = criterion(out, gt_labels[i])
        dy_dx = torch.autograd.grad(y, nets[i].parameters())
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        original_dy_dxs.append(original_dy_dx)

        # predict the ground-truth label
        label_preds.append(torch.argmin(torch.sum(original_dy_dx[-2], dim=-1) , dim=-1).detach().reshape((1,)).requires_grad_(False))

    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data, ], lr = args.lr)

    history = []
    history_iters = []
    losses = []
    train_iters = []
    results = []
    args.logger.info('lr = #{}', args.lr)
    for iters in range(Iteration):

        def closure():
            optimizer.zero_grad()
            single_alpha = torch.FloatTensor([0,1])
            # _ = torch.randint(1, (1,))[0]
            _ = torch.rand(1)
            random_alpha = torch.FloatTensor([_, 1 - _])
            # random_alpha = [0.4, 0.3, 0.3]

            # new
            dummy_dy_dxs = []
            for i in range(args.num_servers):
                pred = (nets[i](dummy_data))
                dummy_loss = criterion(pred, label_preds[i])
                dummy_dy_dxs.append(torch.autograd.grad(dummy_loss, nets[i].parameters(), create_graph=True))
            losses = []
            for i in range(args.num_servers):
                _loss = 0
                for gx, gy in zip(dummy_dy_dxs[i], original_dy_dxs[i]):
                    _loss += ((gx - gy) ** 2).sum()
                losses.append(_loss)



            # add class num into weights
            class_list = [5749, 2]
            # game_alpha = np.multiply(game_alpha, class_list)

            if args.diff_task_agg == 'game':
                game = NashMSFL(n_tasks=args.num_servers)
                _, _, game_alpha = game.get_weighted_loss(losses=losses, dummy_data=dummy_data)
                game_alpha = [game_alpha[i] / sum(game_alpha) for i in range(len(game_alpha))]
                grad_diff = sum([losses[i] * game_alpha[i] for i in range(len(game_alpha))])
            elif args.diff_task_agg == 'single':
                grad_diff = sum([losses[i] * single_alpha[i] for i in range(len(single_alpha))])
            elif args.diff_task_agg == 'random':
                grad_diff = sum([losses[i] * random_alpha[i] for i in range(len(random_alpha))])

            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        result = save_eval(args.get_eval_metrics(), dummy_data, gt_data)
        args.logger.info('loss: #{}, mse: #{}, lpips: #{}, psnr: #{}, ssim: #{}', losses[-1], result[0], result[1], result[2], result[3])
        results.append(result)

        if args.earlystop > 0:
            _es = early_stop(args, current_loss, iters, tp, final_img, dummy_data, imidx_list, imidx, num_dummy,
                             save_path)
            if _es:
                break

        # save the training image
        if iters % int(Iteration / args.log_interval) == 0:
            save_img(iters, args, history, tp, dummy_data, num_dummy, history_iters, gt_data, save_path, imidx_list,
                     str_time)

        # save the final image
        if args.save_final_img:
            save_final_img(iters, final_iter, final_img, tp, dummy_data, imidx, num_dummy, save_path, args, imidx_list)

    label = torch.argmax(dummy_label, dim=-1).detach().item()
    args.logger.info("inversion finished")

    return imidx_list, final_iter, final_img, results

def mdlg_mt_old(args, device, num_dummy, idx_shuffle, tt, tp, dst, nets, num_classes, Iteration, save_path, str_time):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1
    net = nets[0]
    net_1 = nets[1]

    for imidx in range(num_dummy):
        idx, imidx_list = set_idx(imidx, imidx_list, idx_shuffle)
        tmp_datum = tt(dst[idx][0]).float().to(device)
        tmp_datum = tmp_datum.view(1, *tmp_datum.size())
        tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)

        '''get new mapping label for the same data sample'''
        tmp_label_1 = label_mapping(origin_label = dst[idx][1]).to(device)
        # tmp_label_1 = torch.Tensor([dst[idx][1]]).long().to(device)

        tmp_label = tmp_label.view(1, )
        tmp_label_1 = tmp_label_1.view(1, )

        if imidx == 0:
            gt_data = tmp_datum
            gt_label = tmp_label
            gt_label_1 = tmp_label_1
        else:
            gt_data = torch.cat((gt_data, tmp_datum), dim=0)
            gt_label = torch.cat((gt_label, tmp_label), dim=0)
            gt_label_1 = torch.cat((gt_label_1, tmp_label_1), dim=0)


    # compute original gradient

    out = net(gt_data)
    out_1 = net_1(gt_data)
    y = criterion(out, gt_label)
    y_1 = criterion(out_1, gt_label_1)
    dy_dx = torch.autograd.grad(y, net.parameters())
    dy_dx_1 = torch.autograd.grad(y_1, net_1.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    original_dy_dx_1 = list((_.detach().clone() for _ in dy_dx_1))

    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data, ], lr = args.lr)

    # predict the ground-truth label
    label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1) , dim=-1).detach().reshape(
        (1,)).requires_grad_(False)
    label_pred_1 = torch.argmin(torch.sum(original_dy_dx_1[-2], dim=-1), dim=-1).detach().reshape(
        (1,)).requires_grad_(False)    # predict the ground-truth label
    history = []
    history_iters = []
    losses = []
    train_iters = []
    results = []
    args.logger.info('lr = #{}', args.lr)
    for iters in range(Iteration):
        result = []

        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            pred_1 = net_1(dummy_data)
            dummy_loss = criterion(pred, label_pred)
            dummy_loss_1 = criterion(pred_1, label_pred_1)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            dummy_dy_dx_1 = torch.autograd.grad(dummy_loss_1, net_1.parameters(), create_graph=True)
            alpha = torch.FloatTensor([0.99,0.01])
            dummy_dy_dx_list = [dummy_dy_dx, dummy_dy_dx_1]
            original_dy_dx_list = [original_dy_dx, original_dy_dx_1]
            losses = []

            for i in range(len(dummy_dy_dx_list)):
                _loss = 0
                for gx, gy in zip(dummy_dy_dx_list[i], original_dy_dx_list[i]):
                    _loss += ((gx - gy) ** 2).sum()
                losses.append(_loss)

            # in this file
            # game_alpha = get_weighted_loss(losses= losses, dummy_data= dummy_data)
            # game_alpha = [game_alpha[i]/sum(game_alpha) for i in range(len(game_alpha))]

            game = NashMSFL()
            _, _, game_alpha = game.get_weighted_loss(losses= losses, dummy_data= dummy_data)
            game_alpha = [game_alpha[i]/sum(game_alpha) for i in range(len(game_alpha))]

            if args.use_game == True:
                grad_diff = sum([losses[i] * game_alpha[i] for i in range(len(game_alpha))])
            else:
                grad_diff = sum([losses[i] * alpha[i] for i in range(len(alpha))])

            grad_diff.backward()
            return grad_diff
        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        result = save_eval(args.get_eval_metrics(), dummy_data, gt_data)
        args.logger.info('loss: #{}, mse: #{}, lpips: #{}, psnr: #{}, ssim: #{}', losses[-1], result[0], result[1], result[2], result[3])
        results.append(result)

        if args.earlystop > 0:
            _es = early_stop(args, current_loss, iters, tp, final_img, dummy_data, imidx_list, imidx, num_dummy,
                             save_path)
            if _es:
                break

        # save the training image
        if iters % int(Iteration / args.log_interval) == 0:
            save_img(iters, args, history, tp, dummy_data, num_dummy, history_iters, gt_data, save_path, imidx_list,
                     str_time)

        # save the final image
        if args.save_final_img:
            save_final_img(iters, final_iter, final_img, tp, dummy_data, imidx, num_dummy, save_path, args, imidx_list)

    label = torch.argmax(dummy_label, dim=-1).detach().item()
    args.logger.info("inversion finished")

    return imidx_list, final_iter, final_img, results


def cpa(args, device, num_dummy, idx_shuffle, tt, tp, dst, net, num_classes, Iteration, save_path):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1
    lr = 0.0001

    for imidx in range(num_dummy):
        idx, imidx_list = set_idx(imidx, imidx_list, idx_shuffle)

        tmp_datum = tt(dst[idx][0]).float().to(device)
        tmp_datum = tmp_datum.view(1, *tmp_datum.size())
        tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
        tmp_label = tmp_label.view(1, )
        if imidx == 0:
            gt_data = tmp_datum
            gt_label = tmp_label
        else:
            gt_data = torch.cat((gt_data, tmp_datum), dim=0)
            gt_label = torch.cat((gt_label, tmp_label), dim=0)

    out = net(gt_data)
    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr= lr)
    history = []
    history_iters = []
    losses = []
    mses = []
    train_iters = []
    args.logger.info('lr = #{}', args.lr)
    for iters in range(Iteration):

        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            dummy_loss = - torch.mean(torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=-1))
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff
        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

        # if abs(current_loss) < args.get_earlystop():  # converge
        #     final_iter = iters
        #     print('this is final iter')
        #     final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
        #     break

        if iters % int(Iteration / args.log_interval) == 0:
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            args.logger.info('current time: #{}, current iter #{}, loss = #{}, mse = #{}', current_time, iters,
                             current_loss, mses[-1])
            history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            history_iters.append(iters)

            for imidx in range(num_dummy):
                plt.figure(figsize=(12, 8))
                plt.subplot(3, 10, 1)
                plt.imshow(tp(gt_data[imidx].cpu()))
                for i in range(min(len(history), 29)):
                    plt.subplot(3, 10, i + 2)
                    plt.imshow(history[i][imidx])
                    plt.title('iter=%d' % (history_iters[i]))
                    plt.axis('off')
                plt.savefig('%s/CPA_on_%s_%05d_%s_%s_%s.png' % (save_path, imidx_list, imidx_list[imidx], args.get_dataset(),args.get_net(), str(int(time.time())) ))
                plt.close()

        elif iters == final_iter:
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])

    loss = losses
    label = torch.argmax(dummy_label, dim=-1).detach().item()
    mse = mses
    args.logger.info('loss of CPA: #{}, mse of CPA: #{}, label of CPA: #{}', loss[-1], mse[-1], label)
    args.logger.info('final iter: #{}', final_iter)
    return imidx_list, final_iter, final_img
