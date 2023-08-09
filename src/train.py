import os
import time
import shutil
import argparse
from torch.utils.tensorboard import SummaryWriter
from networks.cascadenet import *
from dataloader.dataloader import get_dataloader


def get_parser():
    parser = argparse.ArgumentParser(description='Training CMRxRecon model.')
    parser.add_argument('--trn_dir', type=str,
                        help='Directory of training data.')
    parser.add_argument('--val_dir', type=str,
                        help='Directory of validation data.')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='Data sample ratio.')
    parser.add_argument('--net_type', type=str,
                        choices=['resnet', 'unet'], default='resnet',
                        help='Type of network, should be resnet or unet.')
    parser.add_argument('--dimensions', type=str,
                        choices=['2', '3', '2+1'], default='2+1',
                        help='Dimension of network, should be 2 or 3 or 2+1.')
    parser.add_argument('--real', default=False, action='store_true',
                        help='Real-valued or complex-valued network.')
    parser.add_argument('--lamda', type=float, default=0.01,
                        help='Data consistency regularization parameter.')
    parser.add_argument('--net_depth', type=int, default=3,
                        help='Depth of network.')
    parser.add_argument('--cascade_depth', type=int, default=3,
                        help='Number of data consistency blocks.')
    parser.add_argument('--num_filters', type=int, default=64,
                        help='Number of filters of intermediate layers.')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Kernel size of convolutional layers.')
    parser.add_argument('--bias', default=False, action='store_true',
                        help='Add bias or not for convolutional layers.')
    parser.add_argument('--normalization', type=str,
                        choices=[None, 'batch', 'instance'], default=None,
                        help='Type of normalization, should be None or '
                             'batch or instance.')
    parser.add_argument('--activation', type=str,
                        choices=['ReLU', 'LeakyReLU'], default='ReLU',
                        help='Type of activation, should be ReLU '
                             'or LeakyReLU')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate.')
    parser.add_argument('--use_gpu', default=False, action='store_true',
                        help='Use GPU for training.')
    return parser


def get_net(net_type, dimensions, real, multi_coil, lamda,
            net_depth, cascade_depth, num_filters,
            kernel_size, bias, normalization, activation):
    if net_type == 'unet':
        if dimensions == '2':
            if real:
                net = MultiCoilCascadeUNet2d if multi_coil \
                    else SingleCoilCascadeUNet2d
            else:
                net = MultiCoilCascadeCUNet2d if multi_coil \
                    else SingleCoilCascadeCUNet2d
        elif dimensions == '3':
            if real:
                net = MultiCoilCascadeUNet3d if multi_coil \
                    else SingleCoilCascadeUNet3d
            else:
                net = MultiCoilCascadeCUNet3d if multi_coil \
                    else SingleCoilCascadeCUNet3d
        elif dimensions == '2+1':
            if real:
                net = MultiCoilCascadeUNet2plus1d if multi_coil \
                    else SingleCoilCascadeUNet2plus1d
            else:
                net = MultiCoilCascadeCUNet2plus1d if multi_coil \
                    else SingleCoilCascadeCUNet2plus1d
        else:
            raise ValueError('dimensions should be 2 or 3 or 2+1.')
    elif net_type == 'resnet':
        if dimensions == '2':
            if real:
                net = MultiCoilCascadeResNet2d if multi_coil \
                    else SingleCoilCascadeResNet2d
            else:
                net = MultiCoilCascadeCResNet2d if multi_coil \
                    else SingleCoilCascadeCResNet2d
        elif dimensions == '3':
            if real:
                net = MultiCoilCascadeResNet3d if multi_coil \
                    else SingleCoilCascadeResNet3d
            else:
                net = MultiCoilCascadeCResNet3d if multi_coil \
                    else SingleCoilCascadeCResNet3d
        elif dimensions == '2+1':
            if real:
                net = MultiCoilCascadeResNet2plus1d if multi_coil \
                    else SingleCoilCascadeResNet2plus1d
            else:
                net = MultiCoilCascadeCResNet2plus1d if multi_coil \
                    else SingleCoilCascadeCResNet2plus1d
        else:
            raise ValueError('dimensions should be 2 or 3 or 2+1.')
    else:
        raise ValueError('net_type should be unet or resnet.')
    return net(lamda, net_depth, int(real) + 1, int(real) + 1,
               cascade_depth, num_filters, kernel_size, bias,
               normalization, activation)


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
    net.train()
    avg_loss = 0.0
    global_step = epoch * len(dataloader)

    start_time_epoch = start_time_step = time.perf_counter()
    for step, item in enumerate(dataloader):
        imsub = item['imsub'].to('cuda') if use_gpu else item['imsub']
        ksub = item['ksub'].to('cuda') if use_gpu else item['ksub']
        mask = item['mask'].to('cuda') if use_gpu else item['mask']
        imfull = item['imfull'].to('cuda') if use_gpu else item['imfull']
        if 'sens' in item:
            sens = item['sens'].to('cuda') if use_gpu else item['sens']
            imrecon = net(imsub, mask, ksub, sens)
        else:
            imrecon = net(imsub, mask, ksub)
        loss = torch.nn.functional.l1_loss(imrecon, imfull)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    net.eval()
    avg_loss = 0.0

    start_time = time.perf_counter()
    with torch.no_grad():
        for idx, item in enumerate(dataloader):
            imsub = item['imsub'].to('cuda') if use_gpu else item['imsub']
            ksub = item['ksub'].to('cuda') if use_gpu else item['ksub']
            mask = item['mask'].to('cuda') if use_gpu else item['mask']
            imfull = item['imfull'].to('cuda') if use_gpu else item['imfull']
            if 'sens' in item:
                sens = item['sens'].to('cuda') if use_gpu else item['sens']
                imrecon = net(imsub, mask, ksub, sens)
            else:
                imrecon = net(imsub, mask, ksub)
            avg_loss += torch.nn.functional.l1_loss(imrecon, imfull)

            if idx % (len(dataloader) // 6) == 0:
                display(writer, imsub, imfull, imrecon, epoch, idx)

        avg_loss /= len(dataloader)
        writer.add_scalar('val_loss', avg_loss, epoch)
        val_time = time.perf_counter() - start_time
    return avg_loss, val_time


def display(writer, imsub, imfull, imrecon, epoch, idx):
    if len(imsub.shape) == 4:
        imsub = imsub[0]
        imfull = imfull[0]
        imrecon = imrecon[0]
    else:
        imsub = imsub[0, :, 0]
        imfull = imfull[0, :, 0]
        imrecon = imrecon[0, :, 0]
    if imsub.shape[0] == 1:
        imsub = imsub[0]
        imfull = imfull[0]
        imrecon = imrecon[0]
    else:
        imsub = torch.complex(imsub.select(0, 0), imsub.select(0, 1))
        imfull = torch.complex(imfull.select(0, 0), imfull.select(0, 1))
        imrecon = torch.complex(imrecon.select(0, 0), imrecon.select(0, 1))
    imsub = torch.abs(imsub)
    imfull = torch.abs(imfull)
    imrecon = torch.abs(imrecon)
    im = torch.vstack((imsub, imfull, imrecon))
    writer.add_images(f'sample {idx}', im, global_step=epoch,
                      dataformats='HW')


def main():
    args = get_parser().parse_args()
    trn_dir = args.trn_dir
    val_dir = args.val_dir
    sample_ratio = args.sample_ratio
    net_type = args.net_type
    dimensions = args.dimensions
    real = args.real
    multi_coil = 'MultiCoil' in trn_dir
    lamda = args.lamda
    net_depth = args.net_depth
    cascade_depth = args.cascade_depth
    num_filters = args.num_filters
    kernel_size = args.kernel_size
    bias = args.bias
    normalization = args.normalization
    activation = args.activation
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    use_gpu = args.use_gpu

    net_name = \
        ['multi' if multi_coil else 'single', net_type,
         dimensions + 'd', str(lamda), str(net_depth),
         str(cascade_depth), str(num_filters), str(kernel_size)]
    if not real:
        net_name = ['complex'] + net_name
    if bias:
        net_name.append('bias')
    if normalization:
        net_name.append(normalization)
    if activation:
        net_name.append(activation)
    net_name = '_'.join(net_name)
    out_dir = os.path.join('..', 'experiments', net_name)
    log_dir = os.path.join(out_dir, 'log')
    ckpt_filename = os.path.join(out_dir, 'ckpt.pt')
    best_ckpt_filename = os.path.join(out_dir, 'best_ckpt.pt')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    net = get_net(net_type, dimensions, real, multi_coil, lamda,
                  net_depth, cascade_depth, num_filters,
                  kernel_size, bias, normalization, activation)
    if use_gpu:
        net = net.to('cuda')
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    trn_dataloader = get_dataloader(trn_dir, True, dimensions,
                                    real, multi_coil, sample_ratio)
    val_dataloader = get_dataloader(val_dir, False, dimensions,
                                    real, multi_coil, sample_ratio)
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
