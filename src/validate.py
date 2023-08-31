import os
import time
import argparse
from networks.models import *
from utils.io import *


def get_parser():
    parser = argparse.ArgumentParser(
        description='Deployment of CMRxRecon model.')
    parser.add_argument('--val_dir', type=str,
                        help='Directory of deployment data.')
    parser.add_argument('--undersample_ratio', type=int,
                        help='Undersampling ratio.')
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
    parser.add_argument('--use_gpu', default=False, action='store_true',
                        help='Use GPU for deployment.')
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


def load_ckpt(ckpt_filename, net):
    ckpt = torch.load(ckpt_filename)
    net.load_state_dict(ckpt['net'])
    return net


def preprocess(item, center_lines, subsample_ratio):
    item['kfull'] = torch.from_numpy(item['kfull'])
    if 'sens' in item:
        item['sens'] = torch.from_numpy(item['sens'])
    mask = torch.zeros_like(item['kfull'], dtype=torch.float)
    center_start = (item['kfull'].shape[-2] - center_lines) // 2
    center_end = center_start + center_lines
    mask[..., center_start:center_end, :] = 1
    mask[..., ::subsample_ratio, :] = 1
    item['mask'] = mask
    item['ksub'] = mask * item['kfull']
    item['ksub'] = torch.unsqueeze(torch.unsqueeze(
        item['ksub'], dim=0), dim=0)
    item['mask'] = torch.unsqueeze(torch.unsqueeze(
        item['mask'], dim=0), dim=0)
    if 'sens' in item:
        item['sens'] = torch.unsqueeze(torch.unsqueeze(
            item['sens'], dim=0), dim=0)
    return item


def deploy(net, item, use_gpu):
    with torch.no_grad():
        ksub = item['ksub'].to('cuda') if use_gpu else item['ksub']
        mask = item['mask'].to('cuda') if use_gpu else item['mask']
        if 'sens' in item:
            sens = item['sens'].to('cuda') if use_gpu else item['sens']
            imcnn = net(ksub, mask, sens)
        else:
            imcnn = net(ksub, mask)
    return imcnn.to('cpu') if use_gpu else imcnn


def postprocess(item):
    item = torch.squeeze(item)
    item = item.numpy()
    return item


def main():
    args = get_parser().parse_args()
    val_dir = args.val_dir
    undersample_ratio = args.undersample_ratio
    real = args.real
    multi_coil = 'MultiCoil' in val_dir
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
    exp_dir = os.path.join('..', 'experiments', net_name)
    out_dir = os.path.join(exp_dir, 'ValidationSet',
                           f'AccFactor{undersample_ratio:02d}')
    ckpt_filename = os.path.join(exp_dir, 'best_ckpt.pt')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    net = get_net(real, multi_coil, cascade_depth, lamda,
                  net_depth, num_filters, kernel_size, bias,
                  normalization, activation, down, up)
    if use_gpu:
        net = net.to('cuda')
    net = load_ckpt(ckpt_filename, net)

    filenames = [filename for filename in os.listdir(val_dir)
                 if filename.endswith('.h5')]
    for filename in filenames:
        start_time = time.perf_counter()
        kspace = h5read(os.path.join(val_dir, filename), 'kspace')
        if multi_coil:
            sens = h5read(os.path.join(val_dir, filename), 'sensitivity_map')
        num_slices = kspace.shape[0]
        imcnn = np.zeros((num_slices,) + kspace.shape[-2:])
        for i in range(num_slices):
            item = {'kfull': kspace[i]}
            if multi_coil:
                item['sens'] = sens[i]
            item = preprocess(item, 24, undersample_ratio)
            item = deploy(net, item, use_gpu)
            item = postprocess(item)
            imcnn[i] = item
        print(f'[o] Deployment time: '
              f'{time.perf_counter() - start_time:.2f} s')
        h5write(os.path.join(out_dir, filename), 'imcnn', imcnn)


if __name__ == '__main__':
    main()
