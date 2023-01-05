from __future__ import print_function
import argparse
import torch
from torchvision import transforms
import scipy.io as sio
import logging
from utils import *
import sys

# our stuff
from train import iou_shapelayer, name2dataset, name2net, net_default, dataset_default
from voxel2layer_torch import *
from ResNet import *
from DatasetLoader import *
from DatasetCollector import *


def main(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
 
    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")

    torch.manual_seed(1)
    
    savegame = torch.load(args.file)
    args.side = savegame['side']
    id1, id2, id3 = generate_indices(args.side, device)
    
    # load dataset
    try:
        logging.info(f'Initializing dataset "{args.dataset}"')
        Collector = name2dataset[args.dataset](resolution=args.side, base_dir=args.basedir, shapenet_base_dir=args.shapenet_base_dir)       
    except KeyError:
        logging.error(f'A dataset named "{args.dataset}" is not available.')
        exit(1)

    logging.info('Initializing dataset loader')
    if args.set == 'val':
        samples = Collector.val()
    elif args.set == 'test':
        samples = Collector.test()

    num_samples = len(samples)
    logging.info(f'Found {num_samples} test samples.')
    test_loader = torch.utils.data.DataLoader(DatasetLoader(samples, args.ncomp,
        input_transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=args.batchsize, shuffle=False, num_workers=args.nthreads,
        pin_memory=True
    )
    samples = []

    net = name2net[args.net](
        num_input_channels=3, 
        num_initial_channels=savegame['ninf'],
        num_inner_channels=savegame['ngf'],
        num_penultimate_channels=savegame['noutf'], 
        num_output_channels=6*args.ncomp,
        input_resolution=128, 
        output_resolution=savegame['side'],
        num_downsampling=savegame['down'], 
        num_blocks=savegame['block']
    ).to(device)
    logging.info(net)
    net.load_state_dict(savegame['state_dict'])
    
    net.eval()
   
    # Create results folder
    os.makedirs(args.save_results, exist_ok=True)

    agg_iou   = 0.
    count     = 0
    results   = torch.zeros(args.batchsize*100, 6*args.ncomp, savegame['side'],savegame['side']).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            pred    = net(inputs)
            iou, bs = iou_shapelayer(shlx2shl(pred), targets, id1, id2, id3)
            agg_iou += float(iou)
            count   += bs
            logging.info(f'{batch_idx}: {count}/{num_samples} Mean IoU = {round(100 * agg_iou / count,2)}')
            i = batch_idx % 100
            results[i*args.batchsize:i*args.batchsize+bs,:,:,:] = pred
            if args.save_iter != -1 and i == args.save_iter:
                sio.savemat(
                    f'{args.save_results}/b_{str(batch_idx//100).zfill(3)}.mat', 
                    { 'results': results.detach().cpu().numpy() }, 
                    do_compression=True
                )
                saved = True
            if i == 0:
                saved = False
        
        if args.save_iter != -1 and not saved:
            results = results[:i*args.batchsize+bs,:,:,:]
            sio.savemat(
                f'b_{str(batch_idx//100).zfill(3)}.mat', 
                { 'results': results.detach().cpu().numpy() }, 
                do_compression=True
            )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info(sys.argv)

    parser = argparse.ArgumentParser(description='Train a Matryoshka Network')

    # general options
    parser.add_argument('--title',        type=str,            default='matryoshka',  help='Title in logs, filename (default: matryoshka).')
    parser.add_argument('--no_cuda',      action='store_true', default=False,         help='disables CUDA training')
    parser.add_argument('--gpu',          type=int,            default=0,             help='GPU ID if cuda is available and enabled')
    parser.add_argument('--batchsize',    type=int,            default=32,            help='input batch size for training (default: 128)')
    parser.add_argument('--nthreads',     type=int,            default=4,             help='number of threads for loader')
    parser.add_argument('--save_results', type=str,            default='./results',   help='Folder where predictions will be saved')
    parser.add_argument('--save_iter',    type=str,            default=40,            help='Period and number of samples (save_iter * batchsize) to save. Enter -1 to turn off save')

    # options for dataset
    parser.add_argument('--dataset',          type=str,            default=dataset_default, help=('Dataset [%s]' % ','.join(name2dataset.keys())))
    parser.add_argument('--set',              type=str,            default='val',           help='Validation or test set. (default: val)', choices=['val', 'test'])
    parser.add_argument('--basedir',          type=str,            default='./data/',       help='Base directory for dataset.')
    parser.add_argument('--shapenet_base_dir',type=str,            default='./ShapeNetRendering/', help='Directory with rendered images for shapenet dataset.')
    
    # options for network
    parser.add_argument('--file',  type=str, default=None, help='Savegame')
    parser.add_argument('--net',   type=str, default=net_default, help=('Network architecture [%s]' % ','.join(name2net.keys())))
    parser.add_argument('--ncomp', type=int, default=1,   help='Number of nested shape layers (default: 1)') 

    args = parser.parse_args()
    main(args)

