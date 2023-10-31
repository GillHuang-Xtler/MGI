import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt


def dlg(args, device, num_dummy, idx_shuffle, tt, tp, dst, net, num_classes, Iteration, save_path):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1

    for imidx in range(num_dummy):
        idx = idx_shuffle[imidx]
        imidx_list.append(idx)
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

        if abs(current_loss) < 1e-5:  # converge
            final_iter = iters
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            # plt.imshow(final_img[0][imidx])
            # plt.title('iter=%d' % (final_iter))
            # plt.axis('off')
            # plt.savefig('%s/DLG_final_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
            # plt.close()
            break

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
                plt.savefig('%s/DLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
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
    args.logger.info('loss of DLG: #{}, mse of DLG: #{}, label of DLG: #{}', loss[-1], mse[-1], label)
    args.logger.info('final iter: #{}', final_iter)

    return imidx_list, final_iter, final_img

def idlg(args, device, num_dummy, idx_shuffle, tt, tp, dst, net, num_classes, Iteration, save_path):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1

    for imidx in range(num_dummy):
        idx = idx_shuffle[imidx]
        imidx_list.append(idx)
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
    train_iters = []
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
        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

        if abs(current_loss) < 1e-5:  # converge
            final_iter = iters
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            break

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
                plt.savefig('%s/iDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                plt.close()
        elif iters == final_iter:
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])


    loss = losses
    label = torch.argmax(dummy_label, dim=-1).detach().item()
    mse = mses
    args.logger.info('loss of iDLG: #{}, mse of iDLG: #{}, label of iDLG: #{}', loss[-1], mse[-1], label)
    args.logger.info('final iter: #{}', final_iter)
    return imidx_list, final_iter, final_img

def dlgadam(args, device, num_dummy, idx_shuffle, tt, tp, dst, net, num_classes, Iteration, save_path):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1
    lr = 0.1

    for imidx in range(num_dummy):
        idx = idx_shuffle[imidx]
        imidx_list.append(idx)
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

        if abs(current_loss) < 1e-5:  # converge
            final_iter = iters
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            break


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
                plt.savefig('%s/DLGAdam_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                plt.close()
        elif iters == final_iter:
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])

    loss = losses
    label = torch.argmax(dummy_label, dim=-1).detach().item()
    mse = mses
    args.logger.info('loss of DLGAdam: #{}, mse of DLGAdam: #{}, label of DLGAdam: #{}', loss[-1], mse[-1], label)
    args.logger.info('final iter: #{}', final_iter)

    return imidx_list, final_iter, final_img

def invg(args, device, num_dummy, idx_shuffle, tt, tp, dst, net, num_classes, Iteration, save_path):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1
    lr = 0.1

    for imidx in range(num_dummy):
        idx = idx_shuffle[imidx]
        imidx_list.append(idx)
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
                cos = nn.CosineSimilarity(dim = 0)
                grad_diff += (1-cos(gx, gy)).sum()
            grad_diff.backward()
            return grad_diff
        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

        if abs(current_loss) < 1e-5:  # converge
            final_iter = iters
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            break


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
                plt.savefig('%s/InvG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                plt.close()

        elif iters == final_iter:
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])

    loss = losses
    label = torch.argmax(dummy_label, dim=-1).detach().item()
    mse = mses
    args.logger.info('loss of InvG: #{}, mse of InvG: #{}, label of InvG: #{}', loss[-1], mse[-1], label)
    args.logger.info('final iter: #{}', final_iter)
    return imidx_list, final_iter, final_img

def mdlg(args, device, num_dummy, idx_shuffle, tt, tp, dst, net, net_1, num_classes, Iteration, save_path):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1

    for imidx in range(num_dummy):
        idx = idx_shuffle[imidx]
        imidx_list.append(idx)
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
    out = net(gt_data)
    out_1 = net_1(gt_data)
    y = criterion(out, gt_label)
    y_1 = criterion(out_1, gt_label)
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
    train_iters = []
    args.logger.info('lr = #{}', args.lr)
    for iters in range(Iteration):

        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            pred_1 = net_1(dummy_data)
            dummy_loss = criterion(pred, label_pred)
            dummy_loss_1 = criterion(pred_1, label_pred_1)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            dummy_dy_dx_1 = torch.autograd.grad(dummy_loss_1, net_1.parameters(), create_graph=True)
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            for gx_1, gy_1 in zip(dummy_dy_dx_1, original_dy_dx_1):
                grad_diff += ((gx_1 - gy_1) ** 2).sum()
            grad_diff.backward()
            return grad_diff
        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data - gt_data) ** 2).item())

        if abs(current_loss) < 1e-5:  # converge
            final_iter = iters
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            break

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
                plt.savefig('%s/mDLG_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
                plt.close()

        elif iters == final_iter:
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])

    loss = losses
    label = torch.argmax(dummy_label, dim=-1).detach().item()
    mse = mses
    args.logger.info('loss of mDLG: #{}, mse of mDLG: #{}, label of mDLG: #{}', loss[-1], mse[-1], label)
    args.logger.info('final iter: #{}', final_iter)
    return imidx_list, final_iter, final_img

def cpa(args, device, num_dummy, idx_shuffle, tt, tp, dst, net, num_classes, Iteration, save_path):
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1
    lr = 0.0001

    for imidx in range(num_dummy):
        idx = idx_shuffle[imidx]
        imidx_list.append(idx)
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

        if abs(current_loss) < 1e-5:  # converge
            final_iter = iters
            print('this is final iter')
            final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
            break

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
                plt.savefig('%s/CPA_on_%s_%05d.png' % (save_path, imidx_list, imidx_list[imidx]))
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
