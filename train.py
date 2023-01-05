from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import transforms
import PIL
import logging
from utils import *
import sys
import traceback


from voxel2layer_torch import *
from ResNet import *
from DatasetLoader import *
from DatasetCollector import *

torch.backends.cudnn.benchmark = True


def pos_loss(pred, target, num_components=6):
    """ 
    Modified L1-loss, which penalizes background pixels
    only if predictions are closer than 1 to being considered foreground.
    
    """

    fg_loss  = pred.new_zeros(1)
    bg_loss  = pred.new_zeros(1)
    fg_count = 0 # counter for normalization
    bg_count = 0 # counter for normalization
    
    for i in range(num_components):
        mask     = target[:,i,:,:].gt(0).float().detach()
        target_i = target[:,i,:,:]
        pred_i   = pred[:,i,:,:]
        
        # L1 between prediction and target only for foreground
        dist = pred_i-target_i
        l1 = torch.abs(dist)
        l1_masked = l1.mul(mask)
        l1_mean = l1_masked.mean()
        fg_loss  += l1_mean
        fg_count += torch.mean(mask)

        # flip mask => background
        mask = 1-mask

        # L1 for background pixels > -1
        bg_loss  += torch.mean(((pred_i + 1)).clamp(min=0).mul(mask))
        bg_count += torch.mean(mask)

    return fg_loss / max(1, fg_count) + \
           bg_loss / max(1, bg_count)


def iou_voxel(pred, voxel):
    """ 
    Computes intersection over union between two shapes.
    Returns iou summed over batch
    
    """
    bs,_,h,w = pred.size()
    
    inter = pred.mul(voxel).detach()
    union = pred.add(voxel).detach()
    union = union.sub_(inter)
    inter = inter.sum(3).sum(2).sum(1)
    union = union.sum(3).sum(2).sum(1)
    return inter.div(union).sum(), bs
        

def iou_shapelayer(pred, voxel, id1, id2, id3):
    """ 
    Compares prediction and ground truth shape layers using IoU.
    Returns iou summed over batch and number of samples in batch.
    
    """
       
    pred  = pred.detach()
    voxel = voxel.detach()

    bs, _, side, _ = pred.shape
    vp = pred.new_zeros(bs,side,side,side, requires_grad=False)
    vt = pred.new_zeros(bs,side,side,side, requires_grad=False)
    
    for i in range(bs):
        vp[i,:,:,:] = decode_shape(pred[i,:,:,:].short().permute(1,2,0),  id1, id2, id3)
        vt[i,:,:,:] = decode_shape(voxel[i,:,:,:].short().permute(1,2,0), id1, id2, id3)

    return iou_voxel(vp,vt)


dataset_default = 'ShapeNet'
optim_default   = 'adam'
net_default     = 'resnet'

# register networks, datasets, etc.
name2net        = {net_default: ResNet}
name2dataset    = {\
#    'SanityCheck':SanityCollector, \
#    'ShapeNetPTN':ShapeNetPTNCollector, \
    'ShapeNetCars': ShapeNetCarsOGNCollector,
    'Faust': FaustCollector,
    dataset_default: ShapeNet3DR2N2Collector
}
name2optim      = { optim_default: optim.Adam }


def main(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.shuffle_train = not args.no_shuffle_train
    args.shuffle_val   = not args.no_shuffle_val 

    device = torch.device(f"cuda:{args.gpu}" if args.cuda else "cpu")

    id1, id2, id3 = generate_indices(args.side, device)

    torch.manual_seed(1)

    # load dataset
    try:
        logging.info(f'Initializing dataset "{args.dataset}"')
        Collector = name2dataset[args.dataset](resolution=args.side, base_dir=args.basedir, shapenet_base_dir=args.shapenet_base_dir)        
    except KeyError:
        traceback.print_exc()
        logging.error(f'A dataset named "{args.dataset}" is not available.')
        exit(1)

    logging.info('Initializing dataset loader')
    train_samples = Collector.train()
    logging.info(f'Found {len(train_samples)} training samples.')
    train_loader = torch.utils.data.DataLoader(DatasetLoader(train_samples, args.ncomp,
        input_transform=transforms.Compose([transforms.ToTensor(), RandomColorFlip()])),
        batch_size=args.batchsize, shuffle=args.shuffle_train, num_workers=args.nthreads,
        pin_memory=True
    )

    if not args.no_val:
        val_samples = Collector.val()
        logging.info(f'Found {len(val_samples)} validation samples.')
        val_loader = torch.utils.data.DataLoader(DatasetLoader(val_samples, args.ncomp,
            input_transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=args.batchsize, shuffle=args.shuffle_val, num_workers=args.nthreads,
            pin_memory=True
        )

    # load network
    try:
        logging.info(f'Initializing "{args.net}" network')
        net = name2net[args.net](\
            num_input_channels=3, 
            num_initial_channels=args.ninf,
            num_inner_channels=args.ngf,
            num_penultimate_channels=args.noutf, 
            num_output_channels=6*args.ncomp,
            input_resolution=128, 
            output_resolution=args.side,
            num_downsampling=args.down, 
            num_blocks=args.block
            ).to(device)
        # TODO: Train with multiple gpus
        #net.set_data_parallel(True)
        #net = torch.nn.DataParallel(net, device_ids=args.gpu)
        logging.info(net)
    except KeyError:
        logging.error(f'A network named "{args.net}" is not available.')
        exit(2)

    if args.file:
        savegame = torch.load(args.file)
        net.load_state_dict(savegame['state_dict'])

    # init optimizer
    try:
        logging.info(f'Initializing "{args.optim}" optimizer with learning rate = {args.lr} and weight decay = {args.decay}')
        optimizer = name2optim[args.optim](net.parameters(), lr=args.lr, weight_decay=args.decay)
    except KeyError:
        logging.error(f'An optimizer named "{args.optim}" is not available.')
        exit(3)
   
    # Create results folder
    os.makedirs(args.save_results, exist_ok=True)

    try:
        net.train()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.drop, gamma=0.5)
        agg_loss  = 0.
        count     = 0
        val_results = []

        for epoch in range(1, args.epochs + 1):
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()

                inputs  = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                pred    = net(inputs)
                loss    = pos_loss(pred, shl2shlx(targets), num_components=6*args.ncomp)
                
                loss.backward()        
                optimizer.step()
                agg_loss += loss.detach()
                count    += inputs.shape[0]
                if batch_idx % args.log_inter == 0:
                    logging.info(f'{epoch}/{batch_idx}: Train loss: {str(round(agg_loss.item()/count, 5))} {args.title}')
                    agg_loss = 0.
                    count    = 0
            scheduler.step()
            
            if not args.no_save and epoch % args.save_inter == 0:
                filename = f'{args.save_results}/{args.title}_{args.dataset}_{epoch}.pth.tar'
                logging.info(f'Saving model to {filename}.')
                net.eval()
                torch.save(
                    {
                        'state_dict': net.state_dict(), 
                        'optimizer' : optimizer.state_dict(),
                        'ninf':args.ninf,
                        'ngf':args.ngf,
                        'noutf':args.noutf,
                        'block':args.block,
                        'side': args.side,
                        'down':args.down,
                        'epoch': epoch,
                        'optim': args.optim,
                        'lr': args.lr,
                     }, filename
                )
                net.train()
            # validation
            if not args.no_val and epoch % args.val_inter == 0:
                net.eval()
                agg_iou = 0.
                count   = 0
                with torch.no_grad():    
                    for batch_idx, (inputs, targets) in enumerate(val_loader):

                        inputs  = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                
                        pred     = net(inputs)                
                        iou, bs  = iou_shapelayer(shlx2shl(pred), targets, id1, id2, id3)
                        agg_iou += float(iou)
                        count   += bs
            
                net.train()
                total_iou = (100 * agg_iou / count) if count > 0 else 0
                val_results.append(total_iou)
                logging.info(f'{epoch}: Val set accuracy, iou: {round(total_iou, 2)} {args.title}')                

    except KeyboardInterrupt:
        pass
    finally:
        if len(val_results) != 0:
            np.save(f'{args.save_results}/val_iou.npy', np.asarray(val_results))

    if not args.no_save:
        filename = f'{args.save_results}/{args.title}_{args.dataset}_{epoch}.pth.tar'
        logging.info(f'Saving model to {filename}.')
        torch.save(
            {
                'state_dict': net.state_dict(), 
                'optimizer' : optimizer.state_dict(),
                'ninf':args.ninf,
                'ngf':args.ngf,
                'noutf':args.noutf,
                'block':args.block,
                'side': args.side,
                'down':args.down,
                'epoch': epoch,
                'optim': args.optim,
                'lr': args.lr,
             }, filename
        )
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info(sys.argv)

    parser = argparse.ArgumentParser(description='Train a Matryoshka Network')

    # general options
    parser.add_argument('--title',        type=str,            default='matryoshka',   help='Title in logs, filename (default: matryoshka).')
    parser.add_argument('--no_cuda',      action='store_true', default=False,          help='disables CUDA training')
    parser.add_argument('--gpu',          type=int,            default=0,              help='GPU ID if cuda is available and enabled')
    parser.add_argument('--no_save',      action='store_true', default=False,          help='Disables saving of final model')
    parser.add_argument('--no_val',       action='store_true', default=False,          help='Disable validation for faster training')
    parser.add_argument('--batchsize',    type=int,            default=32,             help='input batch size for training (default: 32)')
    parser.add_argument('--epochs',       type=int,            default=40,             help='number of epochs to train')
    parser.add_argument('--nthreads',     type=int,            default=4,              help='number of threads for loader')
    parser.add_argument('--seed',         type=int,            default=42,             help='random seed (default: 42)')
    parser.add_argument('--val_inter',    type=int,            default=1,              help='Validation interval in epochs (default: 1)')
    parser.add_argument('--log_inter',    type=int,            default=100,            help='Logging interval in batches (default: 100)')
    parser.add_argument('--save_inter',   type=int,            default=10,             help='Saving interval in epochs (default: 10)')
    parser.add_argument('--save_results', type=str,            default='./results',    help='Folder where weights will be saved')

    # options for optimizer
    parser.add_argument('--optim', type=str,   default=optim_default, help=('Optimizer [%s]' % ','.join(name2optim.keys())))
    parser.add_argument('--lr',    type=float, default=1e-3,          help='Learning rate (default: 1e-3)')
    parser.add_argument('--decay', type=float, default=0,             help='Weight decay for optimizer (default: 0)')
    parser.add_argument('--drop',  type=int,   default=30)

    # options for dataset
    parser.add_argument('--dataset',          type=str,            default=dataset_default,        help=('Dataset [%s]' % ','.join(name2dataset.keys())))
    parser.add_argument('--basedir',          type=str,            default='./data/',              help='Base directory for dataset.')
    parser.add_argument('--shapenet_base_dir',type=str,            default='./ShapeNetRendering/', help='Directory with rendered images for shapenet dataset.')
    parser.add_argument('--no_shuffle_train', action='store_true', default=False,                  help='Disable shuffling of training samples')
    parser.add_argument('--no_shuffle_val',   action='store_true', default=False,                  help='Disable shuffling of validation samples')


    # options for network
    parser.add_argument('--file',  type=str, default=None, help='Savegame')
    parser.add_argument('--net',   type=str, default=net_default, help=('Network architecture [%s]' % ','.join(name2net.keys())))
    parser.add_argument('--side',  type=int, default=128, help='Output resolution [if dataset has multiple resolutions.] (default: 128)')
    parser.add_argument('--ncomp', type=int, default=1,   help='Number of nested shape layers (default: 1)')
    parser.add_argument('--ninf',  type=int, default=8,   help='Number of initial feature channels (default: 8)')
    parser.add_argument('--ngf',   type=int, default=512, help='Number of inner channels to train (default: 512)')
    parser.add_argument('--noutf', type=int, default=128, help='Number of penultimate feature channels (default: 128)')
    parser.add_argument('--down',  type=int, default=5,   help='Number of downsampling blocks. (default: 5)')
    parser.add_argument('--block', type=int, default=1,   help='Number of inner blocks at same resolution. (default: 1)')


    args = parser.parse_args()
    main(args)
