import os
import time
import shutil
import argparse

import torch.nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from networks.models import *
from networks.losses import *
from dataloader.dataloader import get_dataloader


def get_parser():
    parser = argparse.ArgumentParser(description='Training CMRxRecon model.')
    parser.add_argument('--trn_dir', type=str,
                        help='Directory of training data.')
    parser.add_argument('--val_dir', type=str,
                        help='Directory of validation data.')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='Data sample ratio.')
    parser.add_argument('--real', default=False, action='store_true',
                        help='Real-valued or complex-valued network.')
    parser.add_argument('--cascade_depth', type=int, default=2,
                        help='Depth of cascade model, 0 or 1 is special, '
                             '0 means k-space domain model,'
                             '1 means image domain model.')
    parser.add_argument('--lamda', type=float, default=0.99,
                        help='Data consistency regularization parameter.')
    parser.add_argument('--net_depth', type=int, default=4,
                        help='Depth of network.')
    parser.add_argument('--num_filters', type=int, default=64,
                        help='Number of filters of intermediate layers.')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Kernel size of convolutional layers.')
    parser.add_argument('--bias', default=False, action='store_true',
                        help='Add bias or not for convolutional layers.')
    parser.add_argument('--normalization', type=str,
                        choices=['instance', 'batch'], default='instance',
                        help='Type of normalization, should be '
                             'instance or batch.')
    parser.add_argument('--activation', type=str,
                        choices=['relu', 'leakyrelu'], default='relu',
                        help='Type of activation, should be '
                             'relu or leakyrelu')
    parser.add_argument('--down', type=str,
                        choices=['avg', 'max'], default='avg',
                        help='Type of downsampling block, should be '
                             'avg or max.')
    parser.add_argument('--up', type=str,
                        choices=['bilinear', 'convtran'], default='bilinear',
                        help='Type of upsampling block, should be '
                             'bilinear or convtran.')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate.')
    parser.add_argument('--use_gpu', default=False, action='store_true',
                        help='Use GPU for training.')
    return parser


def get_net(real, multi_coil, cascade_depth, lamda, net_depth, num_filters,
            kernel_size, bias, normalization, activation, down, up):
    if real:
        if multi_coil:
            if cascade_depth == 0:
                return MultiCoilKspaceDomainNet(
                    2, 2, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up)
            elif cascade_depth == 1:
                return MultiCoilImageDomainNet(
                    2, 2, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up)
            else:
                return MultiCoilCascadeCrossDomainNet(
                    2, 2, cascade_depth, net_depth, num_filters,
                    kernel_size, bias, normalization, activation,
                    down, up, lamda)
        else:
            if cascade_depth == 0:
                return SingleCoilKspaceDomainNet(
                    2, 2, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up)
            elif cascade_depth == 1:
                return SingleCoilImageDomainNet(
                    2, 2, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up)
            else:
                return SingleCoilCascadeCrossDomainNet(
                    2, 2, cascade_depth, net_depth, num_filters,
                    kernel_size, bias, normalization, activation,
                    down, up, lamda)
    else:
        if multi_coil:
            if cascade_depth == 0:
                return MultiCoilComplexKspaceDomainNet(
                    1, 1, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up)
            elif cascade_depth == 1:
                return MultiCoilComplexImageDomainNet(
                    1, 1, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up)
            else:
                return MultiCoilComplexCascadeCrossDomainNet(
                    1, 1, cascade_depth, net_depth, num_filters,
                    kernel_size, bias, normalization, activation,
                    down, up, lamda)
        else:
            if cascade_depth == 0:
                return SingleCoilComplexKspaceDomainNet(
                    1, 1, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up)
            elif cascade_depth == 1:
                return SingleCoilComplexImageDomainNet(
                    1, 1, net_depth, num_filters, kernel_size,
                    bias, normalization, activation, down, up)
            else:
                return SingleCoilComplexCascadeCrossDomainNet(
                    1, 1, cascade_depth, net_depth, num_filters,
                    kernel_size, bias, normalization, activation,
                    down, up, lamda)


def count_parameters(net):
    total_parameters = sum(p.numel() for p in net.parameters())
    trainable_parameters = sum(p.numel() for p in net.parameters()
                               if p.requires_grad)
    return total_parameters, trainable_parameters


def save_ckpt(ckpt_filename, net, optimizer, best_val_loss, epoch):
    torch.save({'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'epoch': epoch}, f=ckpt_filename)


def load_ckpt(ckpt_filename, net, optimizer):
    ckpt = torch.load(ckpt_filename)
    net.load_state_dict(ckpt['net'])
    optimizer.load_state_dict(ckpt['optimizer'])
    best_val_loss = ckpt['best_val_loss']
    start_epoch = ckpt['epoch'] + 1
    return net, optimizer, best_val_loss, start_epoch


def train(net, optimizer, dataloader, writer, epoch, use_gpu):
    start_time_epoch = start_time_step = time.perf_counter()

    net.train()
    avg_loss = 0.0
    global_step = epoch * len(dataloader)
    l1 = torch.nn.L1Loss().to('cuda') if use_gpu else torch.nn.L1Loss()
    ssim = SSIMLoss().to('cuda') if use_gpu else SSIMLoss()
    for step, item in enumerate(dataloader):
        ksub = item['ksub'].to('cuda') if use_gpu else item['ksub']
        mask = item['mask'].to('cuda') if use_gpu else item['mask']
        imfull = item['imfull'].to('cuda') if use_gpu else item['imfull']
        if 'sens' in item:
            sens = item['sens'].to('cuda') if use_gpu else item['sens']
            imcnn = net(ksub, mask, sens)
        else:
            imcnn = net(ksub, mask)
        l1_loss = l1(imcnn, imfull)
        ssim_loss = ssim(imcnn, imfull, imfull.max())
        loss = l1_loss + ssim_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('l1_loss', l1_loss.item(), global_step + step)
        writer.add_scalar('ssim_loss', ssim_loss.item(), global_step + step)
        writer.add_scalar('trn_loss', loss.item(), global_step + step)
        avg_loss = 0.9 * avg_loss + 0.1 * loss.item() \
            if step > 0 else loss.item()
        if (step + 1) % 50 == 0:
            step_time = (time.perf_counter() - start_time_step) / 50
            print(f'[o] Step [{step + 1:4d}/{len(dataloader):4d} | '
                  f'Loss: {loss.item():.4g} | '
                  f'Avg Loss: {avg_loss:.4g} | '
                  f'Avg Time: {step_time:.2f} s')
            start_time_step = time.perf_counter()
    epoch_time = time.perf_counter() - start_time_epoch
    return avg_loss, epoch_time


def validate(net, dataloader, writer, epoch, use_gpu):
    start_time = time.perf_counter()

    net.eval()
    avg_loss = 0.0
    ssim = SSIMLoss().to('cuda') if use_gpu else SSIMLoss()
    display_imfull = []
    display_imcnn = []
    display_mask = []
    with torch.no_grad():
        for idx, item in enumerate(dataloader):
            ksub = item['ksub'].to('cuda') if use_gpu else item['ksub']
            mask = item['mask'].to('cuda') if use_gpu else item['mask']
            imfull = item['imfull'].to('cuda') if use_gpu else item['imfull']
            if 'sens' in item:
                sens = item['sens'].to('cuda') if use_gpu else item['sens']
                imcnn = net(ksub, mask, sens)
                mask = mask[:, :, 0, :, :]
            else:
                imcnn = net(ksub, mask)
            avg_loss += ssim(imcnn, imfull, imfull.max())

            if idx % (len(dataloader) // 16) == 0:
                display_imfull.append(imfull)
                display_imcnn.append(imcnn)
                display_mask.append(mask)

        avg_loss /= len(dataloader)
        writer.add_scalar('val_loss', avg_loss, epoch)
        display(writer, display_imfull, display_imcnn, display_mask, epoch)
        val_time = time.perf_counter() - start_time
    return avg_loss, val_time


def display(writer, display_imfull, display_imcnn, display_mask, epoch):
    max_width = max(imfull.shape[-1] for imfull in display_imfull)
    max_height = max(imfull.shape[-2] for imfull in display_imfull)
    for i in range(len(display_imfull)):
        imfull, imcnn, mask = \
            display_imfull[i], display_imcnn[i], display_mask[i]
        diff_width = max_width - imfull.shape[-1]
        diff_height = max_height - imfull.shape[-2]
        pad = (diff_width // 2, diff_width - diff_width // 2,
               diff_height // 2, diff_height - diff_height // 2)
        display_imfull[i], display_imcnn[i], display_mask[i] = \
            F.pad(imfull, pad), F.pad(imcnn, pad), F.pad(mask, pad)
    display_imfull = torch.cat(display_imfull)
    display_imcnn = torch.cat(display_imcnn)
    display_mask = torch.cat(display_mask)
    writer.add_image('target', make_grid(
        display_imfull, nrow=4, normalize=True, scale_each=True), epoch)
    writer.add_image('output', make_grid(
        display_imcnn, nrow=4, normalize=True, scale_each=True), epoch)
    writer.add_image('mask', make_grid(
        display_mask, nrow=4, normalize=True, scale_each=True), epoch)


def main():
    args = get_parser().parse_args()
    trn_dir = args.trn_dir
    val_dir = args.val_dir
    sample_ratio = args.sample_ratio
    real = args.real
    multi_coil = 'MultiCoil' in trn_dir
    cascade_depth = args.cascade_depth
    lamda = args.lamda
    net_depth = args.net_depth
    num_filters = args.num_filters
    kernel_size = args.kernel_size
    bias = args.bias
    normalization = args.normalization
    activation = args.activation
    down = args.down
    up = args.up
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    use_gpu = args.use_gpu

    net_name = []
    if not real:
        net_name.append('complex')
    net_name.append('multi' if multi_coil else 'single')
    if cascade_depth == 0:
        net_name.append('kspace')
    elif cascade_depth == 1:
        net_name.append('image')
    else:
        net_name.append('cascade' + str(100 * lamda)[:2])
    net_name.append('unet')
    net_name.append(str(net_depth))
    net_name.append(str(num_filters))
    net_name.append(str(kernel_size))
    if bias:
        net_name.append('bias')
    net_name.append(normalization)
    net_name.append(activation)
    net_name.append(down)
    net_name.append(up)
    net_name = '_'.join(net_name)
    out_dir = os.path.join('..', 'experiments', net_name)
    log_dir = os.path.join(out_dir, 'log')
    ckpt_filename = os.path.join(out_dir, 'ckpt.pt')
    best_ckpt_filename = os.path.join(out_dir, 'best_ckpt.pt')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    net = get_net(real, multi_coil, cascade_depth, lamda,
                  net_depth, num_filters, kernel_size, bias,
                  normalization, activation, down, up)
    if use_gpu:
        net = net.to('cuda')
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    trn_dataloader = get_dataloader(trn_dir, True, multi_coil, sample_ratio)
    val_dataloader = get_dataloader(val_dir, False, multi_coil, sample_ratio)
    writer = SummaryWriter(log_dir)

    total_parameters, trainable_parameters = count_parameters(net)
    print(f'[o] Total number of parameters: {total_parameters} | '
          f'Number of trainable parameters: {trainable_parameters}')
    if os.path.exists(ckpt_filename):
        net, optimizer, best_val_loss, start_epoch = \
            load_ckpt(ckpt_filename, net, optimizer)
        num_epochs += start_epoch
    else:
        start_epoch = 0
        best_val_loss = float('Inf')

    for epoch in range(start_epoch, num_epochs):
        trn_loss, trn_time = \
            train(net, optimizer, trn_dataloader, writer, epoch, use_gpu)
        val_loss, val_time = \
            validate(net, val_dataloader, writer, epoch, use_gpu)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_ckpt(ckpt_filename, net, optimizer, best_val_loss, epoch)
            shutil.copyfile(ckpt_filename, best_ckpt_filename)
        else:
            save_ckpt(ckpt_filename, net, optimizer, best_val_loss, epoch)
        print(f'[o] Epoch [{epoch + 1:3d}/{num_epochs:3d}] | '
              f'Training Loss: {trn_loss:.4g} | '
              f'Validation Loss: {val_loss:.4g} | '
              f'Training Time: {trn_time:.2f} s | '
              f'Validation Time: {val_time:.2f} s')

    writer.close()


if __name__ == '__main__':
    main()
