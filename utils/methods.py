import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
from utils.data_processing import set_idx, label_mapping
import numpy as np
import cvxpy as cp
from utils.data_processing import NashMSFL, total_variation
from utils.evaluation import PSNR, SSIM, LPIPS
from torchmetrics.image import TotalVariation


def dlg(args, device, num_dummy, idx_shuffle, tt, tp, dst, net, num_classes, Iteration, save_path):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1

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
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr= args.lr)
    history = []
    history_iters = []
    losses = []
    mses = []
    psnrs = []
    ssims = []
    lpipss = []
    train_iters = []
    results = []
    args.logger.info('lr = #{}', args.lr)
    for iters in range(Iteration):
        result = []
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
        psnrs.append(float(PSNR(dummy_data, gt_data)))
        ssims.append(float(SSIM(dummy_data, gt_data)))
        lpipss.append(float(LPIPS(dummy_data, gt_data)))
        result.append(mses[-1])
        result.append(psnrs[-1])
        result.append(ssims[-1])
        result.append(lpipss[-1])

        results.append(result)

        # if abs(current_loss) < args.get_earlystop():  # converge
        #     final_iter = iters
        #     print('this is final iter')
        #     final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
        #     # plt.imshow(final_img[0][imidx])
        #     # plt.title('iter=%d' % (final_iter))
        #     # plt.axis('off')
        #     # plt.savefig('%s/DLG_final_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
        #     # plt.close()
        #     break

        if iters % int(Iteration / args.log_interval) == 0:
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            args.logger.info('current time: #{}, current iter #{}, loss = #{}, mse = #{}, psnr = #{}, ssim = #{}, lpips = #{}', current_time, iters,
                             current_loss, mses[-1], psnrs[-1], ssims[-1], lpipss[-1])
            history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            history_iters.append(iters)

            for imidx in range(num_dummy):
                plt.figure(figsize=(12, 8))
                plt.subplot(1, 7, 1)
                plt.tight_layout()
                plt.subplots_adjust(wspace=0,hspace=0)
                plt.imshow(tp(gt_data[imidx].cpu()))
                plt.title('original')
                plt.axis('off')
                for i in range(min(len(history), 6)):
                    plt.subplot(1, 7, i + 2)
                    plt.imshow(history[i][imidx])
                    plt.title('iter=%d' % (history_iters[i]))
                    plt.axis('off')
                plt.savefig('%s/DLG_on_%s_%05d_%s_%s.png' % (save_path, imidx_list, imidx_list[imidx], args.get_dataset(),args.get_net()))
                plt.close()
        elif iters == final_iter:
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            # plt.imshow(final_img[0][imidx])
            # plt.title('iter=%d' % (final_iter))
            # plt.axis('off')
            # plt.savefig('%s/DLG_final_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
            # plt.close()

    loss = losses
    label = torch.argmax(dummy_label, dim=-1).detach().item()
    mse = mses
    psnr = psnrs
    ssim = ssims
    lpips = lpipss
    args.logger.info('loss of DLG: #{}, mse of DLG: #{}, label of DLG: #{}, psnr of DLG: #{}, ssim of DLG: #{}, lpips of DLG: #{}', loss[-1], mse[-1], label, psnr[-1], ssim[-1], lpips[-1])
    # args.logger.info('final iter: #{}', final_iter)
    args.logger.info("results : #{}", results)

    return imidx_list, final_iter, final_img, results


def idlg(args, device, num_dummy, idx_shuffle, tt, tp, dst, net, num_classes, Iteration, save_path):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1

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
    optimizer = torch.optim.LBFGS([dummy_data, ], lr=args.lr)

    # predict the ground-truth label
    label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape(
        (1,)).requires_grad_(False)
    history = []
    history_iters = []
    losses = []
    mses = []
    psnrs = []
    ssims = []
    lpipss = []
    train_iters = []
    results = []
    args.logger.info('lr = #{}', args.lr)
    for iters in range(Iteration):
        result = []
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
        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())
        psnrs.append(float(PSNR(dummy_data, gt_data)))
        ssims.append(float(SSIM(dummy_data, gt_data)))
        lpipss.append(float(LPIPS(dummy_data, gt_data)))
        result.append(mses[-1])
        result.append(psnrs[-1])
        result.append(ssims[-1])
        result.append(lpipss[-1])

        results.append(result)

        # if abs(current_loss) < args.get_earlystop():  # converge
        #     final_iter = iters
        #     print('this is final iter')
        #     final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
        #     break

        if iters % int(Iteration / args.log_interval) == 0:
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            args.logger.info('current time: #{}, current iter #{}, loss = #{}, mse = #{}, psnr = #{}, ssim = #{}, lpips = #{}', current_time, iters,
                             current_loss, mses[-1], psnrs[-1], ssims[-1], lpipss[-1])
            history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            history_iters.append(iters)

            for imidx in range(num_dummy):
                plt.figure(figsize=(12, 8))
                plt.subplot(1, 7, 1)
                plt.tight_layout()
                plt.subplots_adjust(wspace=0,hspace=0)
                plt.imshow(tp(gt_data[imidx].cpu()))
                plt.title('original')
                plt.axis('off')
                for i in range(min(len(history), 6)):
                    plt.subplot(1, 7, i + 2)
                    plt.imshow(history[i][imidx])
                    plt.title('iter=%d' % (history_iters[i]))
                    plt.axis('off')
                plt.savefig('%s/iDLG_on_%s_%05d_%s_%s.png' % (save_path, imidx_list, imidx_list[imidx], args.get_dataset(),args.get_net()))
                plt.close()
        elif iters == final_iter:
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])


    loss = losses
    label = torch.argmax(dummy_label, dim=-1).detach().item()
    mse = mses
    psnr = psnrs
    ssim = ssims
    lpips = lpipss
    args.logger.info('loss of iDLG: #{}, mse of iDLG: #{}, label of iDLG: #{}, psnr of iDLG: #{}, ssim of iDLG: #{}, lpips of iDLG: #{}', loss[-1], mse[-1], label, psnr[-1], ssim[-1], lpips[-1])
    # args.logger.info('final iter: #{}', final_iter)
    args.logger.info("results : #{}", results)

    return imidx_list, final_iter, final_img, results


def dlgadam(args, device, num_dummy, idx_shuffle, tt, tp, dst, net, num_classes, Iteration, save_path):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1
    lr = 0.1

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
    optimizer = torch.optim.Adam([dummy_data, dummy_label], lr= lr)
    history = []
    history_iters = []
    losses = []
    mses = []
    psnrs = []
    ssims = []
    lpipss = []
    train_iters = []
    results = []
    args.logger.info('lr = #{}', args.lr)
    for iters in range(Iteration):
        result = []

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
        psnrs.append(float(PSNR(dummy_data, gt_data)))
        ssims.append(float(SSIM(dummy_data, gt_data)))
        lpipss.append(float(LPIPS(dummy_data, gt_data)))
        result.append(mses[-1])
        result.append(psnrs[-1])
        result.append(ssims[-1])
        result.append(lpipss[-1])

        results.append(result)

        if abs(current_loss) < args.get_earlystop():  # converge
            final_iter = iters
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            break


        if iters % int(Iteration / args.log_interval) == 0:
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            args.logger.info('current time: #{}, current iter #{}, loss = #{}, mse = #{}, psnr = #{}, ssim = #{}, lpips = #{}', current_time, iters,
                             current_loss, mses[-1], psnrs[-1], ssims[-1], lpipss[-1])
            history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            history_iters.append(iters)

            for imidx in range(num_dummy):
                plt.figure(figsize=(12, 8))
                plt.subplot(1, 7, 1)
                plt.tight_layout()
                plt.subplots_adjust(wspace=0,hspace=0)
                plt.imshow(tp(gt_data[imidx].cpu()))
                plt.title('original')
                plt.axis('off')
                for i in range(min(len(history), 6)):
                    plt.subplot(1, 7, i + 2)
                    plt.imshow(history[i][imidx])
                    plt.title('iter=%d' % (history_iters[i]))
                    plt.axis('off')
                plt.savefig('%s/DLGAdam_on_%s_%05d_%s_%s.png' % (save_path, imidx_list, imidx_list[imidx], args.get_dataset(),args.get_net()))
                plt.close()
        elif iters == final_iter:
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])

    loss = losses
    label = torch.argmax(dummy_label, dim=-1).detach().item()
    mse = mses
    psnr = psnrs
    ssim = ssims
    lpips = lpipss
    args.logger.info('loss of DLGAdam: #{}, mse of DLGAdam: #{}, label of DLGAdam: #{}, psnr of DLGAdam: #{}, ssim of DLGAdam: #{}, lpips of DLGAdam: #{}', loss[-1], mse[-1], label, psnr[-1], ssim[-1], lpips[-1])
    # args.logger.info('final iter: #{}', final_iter)
    args.logger.info("results : #{}", results)

    return imidx_list, final_iter, final_img, results


def invg(args, device, num_dummy, idx_shuffle, tt, tp, dst, net, num_classes, Iteration, save_path):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1
    lr = 1
    tv = TotalVariation()

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
    psnrs = []
    ssims = []
    lpipss = []
    train_iters = []
    results = []
    args.logger.info('lr = #{}', args.lr)
    for iters in range(Iteration):
        result = []

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
        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())
        psnrs.append(float(PSNR(dummy_data, gt_data)))
        ssims.append(float(SSIM(dummy_data, gt_data)))
        lpipss.append(float(LPIPS(dummy_data, gt_data)))
        result.append(mses[-1])
        result.append(psnrs[-1])
        result.append(ssims[-1])
        result.append(lpipss[-1])

        results.append(result)
        #
        # if abs(current_loss) < args.get_earlystop():  # converge
        #     final_iter = iters
        #     print('this is final iter')
        #     final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
        #     break


        if iters % int(Iteration / args.log_interval) == 0:
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            args.logger.info('current time: #{}, current iter #{}, loss = #{}, mse = #{}, psnr = #{}, ssim = #{}, lpips = #{}', current_time, iters,
                             current_loss, mses[-1], psnrs[-1], ssims[-1], lpipss[-1])
            history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            history_iters.append(iters)

            for imidx in range(num_dummy):
                plt.figure(figsize=(12, 8))
                plt.subplot(1, 7, 1)
                plt.tight_layout()
                plt.subplots_adjust(wspace=0,hspace=0)
                plt.imshow(tp(gt_data[imidx].cpu()))
                plt.title('original')
                plt.axis('off')
                for i in range(min(len(history), 6)):
                    plt.subplot(1, 7, i + 2)
                    plt.imshow(history[i][imidx])
                    plt.title('iter=%d' % (history_iters[i]))
                    plt.axis('off')
                plt.savefig('%s/InvG_on_%s_%05d_%s_%s.png' % (save_path, imidx_list, imidx_list[imidx], args.get_dataset(),args.get_net()))
                plt.close()

        elif iters == final_iter:
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])

    loss = losses
    label = torch.argmax(dummy_label, dim=-1).detach().item()
    mse = mses
    psnr = psnrs
    ssim = ssims
    lpips = lpipss
    args.logger.info('loss of InvG: #{}, mse of InvG: #{}, label of InvG: #{}, psnr of InvG: #{}, ssim of InvG: #{}, lpips of InvG: #{}', loss[-1], mse[-1], label, psnr[-1], ssim[-1], lpips[-1])
    # args.logger.info('final iter: #{}', final_iter)
    args.logger.info("results : #{}", results)

    return imidx_list, final_iter, final_img, results


def mdlg(args, device, num_dummy, idx_shuffle, tt, tp, dst, nets, num_classes, Iteration, save_path):
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
        tmp_label = tmp_label.view(1, )
        if imidx == 0:
            gt_data = tmp_datum
            gt_label = tmp_label
        else:
            gt_data = torch.cat((gt_data, tmp_datum), dim=0)
            gt_label = torch.cat((gt_label, tmp_label), dim=0)



    # compute original gradient

    original_dy_dxs = []
    label_preds = []
    for i in range(args.num_servers):
        out = nets[i](gt_data)
        y = criterion(out, gt_label)
        dy_dx = torch.autograd.grad(y, nets[i].parameters())
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        original_dy_dxs.append(original_dy_dx)

        # predict the ground-truth label
        label_preds.append(torch.argmin(torch.sum(original_dy_dx[-2], dim=-1) , dim=-1).detach().reshape((1,)).requires_grad_(False))
    # old
    # out = net(gt_data)
    # out_1 = net_1(gt_data)
    # y = criterion(out, gt_label)
    # y_1 = criterion(out_1, gt_label)
    # dy_dx = torch.autograd.grad(y, net.parameters())
    # dy_dx_1 = torch.autograd.grad(y_1, net_1.parameters())
    # original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    # original_dy_dx_1 = list((_.detach().clone() for _ in dy_dx_1))
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data, ], lr = args.lr)

    # label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1) , dim=-1).detach().reshape(
    #     (1,)).requires_grad_(False)
    # label_pred_1 = torch.argmin(torch.sum(original_dy_dx_1[-2], dim=-1), dim=-1).detach().reshape(
    #     (1,)).requires_grad_(False)    # predict the ground-truth label



    history = []
    history_iters = []
    losses = []
    mses = []
    psnrs = []
    ssims = []
    lpipss = []
    train_iters = []
    results = []
    args.logger.info('lr = #{}', args.lr)
    for iters in range(Iteration):
        result = []

        def closure():
            optimizer.zero_grad()

            # new
            dummy_dy_dxs = []
            for i in range(args.num_servers):
                pred = (nets[i](dummy_data))
                dummy_loss = criterion(pred, label_preds[i])
                dummy_dy_dxs.append(torch.autograd.grad(dummy_loss, nets[i].parameters(), create_graph=True))

            grad_diff = 0
            for i in range(args.num_servers):
                for gx, gy in zip(dummy_dy_dxs[i], original_dy_dxs[i]):
                    grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

            # pred = net(dummy_data)
            # pred_1 = net_1(dummy_data)
            # dummy_loss = criterion(pred, label_pred)
            # dummy_loss_1 = criterion(pred_1, label_pred_1)
            # dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            # dummy_dy_dx_1 = torch.autograd.grad(dummy_loss_1, net_1.parameters(), create_graph=True)
            # grad_diff = 0
            # for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            #     grad_diff += ((gx - gy) ** 2).sum()
            # for gx_1, gy_1 in zip(dummy_dy_dx_1, original_dy_dx_1):
            #     grad_diff += ((gx_1 - gy_1) ** 2).sum()
            # grad_diff.backward()
            # return grad_diff

        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())
        psnrs.append(float(PSNR(dummy_data, gt_data)))
        ssims.append(float(SSIM(dummy_data, gt_data)))
        lpipss.append(float(LPIPS(dummy_data, gt_data)))
        result.append(mses[-1])
        result.append(psnrs[-1])
        result.append(ssims[-1])
        result.append(lpipss[-1])

        results.append(result)

        # if args.get_earlystop() > 0:
        #     early_stop()
        # if abs(current_loss) < args.get_earlystop():  # converge
        #     final_iter = iters
        #     print('this is final iter')
        #     final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
        #     break

        if iters % int(Iteration / args.log_interval) == 0:
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            args.logger.info('current time: #{}, current iter #{}, loss = #{}, mse = #{}, psnr = #{}, ssim = #{}, lpips = #{}', current_time, iters,
                             current_loss, mses[-1], psnrs[-1], ssims[-1], lpipss[-1])
            history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            history_iters.append(iters)

            for imidx in range(num_dummy):
                plt.figure(figsize=(12, 8))
                plt.subplot(1, 7, 1)
                plt.tight_layout()
                plt.subplots_adjust(wspace=0,hspace=0)
                plt.imshow(tp(gt_data[imidx].cpu()))
                plt.title('original')
                plt.axis('off')
                for i in range(min(len(history), 6)):
                    plt.subplot(1, 7, i + 2)
                    plt.imshow(history[i][imidx])
                    plt.title('iter=%d' % (history_iters[i]))
                    plt.axis('off')
                plt.savefig('%s/mDLG_on_%s_%05d_%s_%s_%s.png' % (save_path, imidx_list, imidx_list[imidx], args.get_dataset(),args.get_net(), str(args.num_servers)))
                plt.close()

        elif iters == final_iter:
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])

    loss = losses
    label = torch.argmax(dummy_label, dim=-1).detach().item()
    mse = mses
    psnr = psnrs
    ssim = ssims
    lpips = lpipss
    args.logger.info('loss of mDLG: #{}, mse of mDLG: #{}, label of mDLG: #{}, psnr of mDLG: #{}, ssim of mDLG: #{}, lpips of mDLG: #{}', loss[-1], mse[-1], label, psnr[-1], ssim[-1], lpips[-1])
    # args.logger.info('final iter: #{}', final_iter)
    args.logger.info("results : #{}", results)

    return imidx_list, final_iter, final_img, results


def mdlg_mt(args, device, num_dummy, idx_shuffle, tt, tp, dst, net, net_1, num_classes, Iteration, save_path):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1

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
    mses = []
    psnrs = []
    ssims = []
    lpipss = []
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
            # print(game_alpha)

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
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())
        psnrs.append(float(PSNR(dummy_data, gt_data)))
        ssims.append(float(SSIM(dummy_data, gt_data)))
        lpipss.append(float(LPIPS(dummy_data, gt_data)))
        result.append(mses[-1])
        result.append(psnrs[-1])
        result.append(ssims[-1])
        result.append(lpipss[-1])

        results.append(result)

        if abs(current_loss) < args.get_earlystop():  # converge
            final_iter = iters
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            break

        if iters % int(Iteration / args.log_interval) == 0:
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            args.logger.info('current time: #{}, current iter #{}, loss = #{}, mse = #{}, psnr = #{}, ssim = #{}, lpips = #{}', current_time, iters,
                             current_loss, mses[-1], psnrs[-1], ssims[-1], lpipss[-1])
            history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            history_iters.append(iters)

            for imidx in range(num_dummy):
                plt.figure(figsize=(12, 8))
                plt.subplot(1, 7, 1)
                plt.tight_layout()
                plt.subplots_adjust(wspace=0,hspace=0)
                plt.imshow(tp(gt_data[imidx].cpu()))
                plt.title('original')
                plt.axis('off')
                for i in range(min(len(history), 6)):
                    plt.subplot(1, 7, i + 2)
                    plt.imshow(history[i][imidx])
                    plt.title('iter=%d' % (history_iters[i]))
                    plt.axis('off')
                plt.savefig('%s/mDLGmt_on_%s_%05d_%s_%s.png' % (save_path, imidx_list, imidx_list[imidx], args.get_dataset(),args.get_net()))
                plt.close()

        elif iters == final_iter:
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])

    loss = losses
    label = torch.argmax(dummy_label, dim=-1).detach().item()
    mse = mses
    psnr = psnrs
    ssim = ssims
    lpips = lpipss
    args.logger.info('loss of mDLGmt: #{}, mse of mDLGmt: #{}, label of mDLGmt: #{}, psnr of mDLGmt: #{}, ssim of mDLGmt: #{}, lpips of mDLGmt: #{}', loss[-1], mse[-1], label, psnr[-1], ssim[-1], lpips[-1])
    # args.logger.info('final iter: #{}', final_iter)
    args.logger.info("results : #{}", results)

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
