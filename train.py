from tensorboardX import SummaryWriter
from dataset.data_loader import data_loader
from models.get_model import get_model
from utils.loss import *
from utils import get_lr, save_checkpoint
from trainer import *
import argparse

def main(opt):
    sw_log = 'logs/%s' % opt.dataset
    sw = SummaryWriter(log_dir=sw_log)
    best_prec1 = 0.

    train_dataset, train_loader, val_dataset, val_loader = data_loader(opt)
    net = get_model(opt, train_dataset.num_classes)

    print(torch.backends.mps.is_available)
    print(torch.backends.mps.is_built)

    device = torch.device("mps" if torch.backends.mps.is_available else"cpu")
    #torch.cuda.set_device(int(opt.gpu_ids[0]))
    use_gpu = torch.backends.mps.is_available() and opt.use_gpu
    net = net.to(device)

    assert opt.optim in ['sgd', 'adam'], 'optim name not found!'
    if opt.optim == 'sgd':
        optimizer = torch.optim.SGD(
            net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optim == 'adam':
        optimizer = torch.optim.Adam(
            net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    assert opt.scheduler in ['plateau','cos','step'], 'scheduler not supported!!!'
    if opt.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    elif opt.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    elif opt.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)

    if opt.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif opt.loss == 'ls':
        criterion = LabelSmoothingLoss(train_dataset.num_classes, opt.smoothing)
    elif opt.loss == 'cuf':
        criterion = CUFLoss(train_dataset.num_classes, opt.smoothing)
    else:
        raise ValueError(f"Unsupported loss function: {opt.loss}")
    if use_gpu:
        criterion = criterion.cuda()

    state = {'model': net, 'train_loader': train_loader,
             'val_loader': val_loader, 'criterion': criterion,
             'config': opt,
             'optimizer': optimizer}

    print(opt)
    if not opt.train:
        prec1, val_loss = validate(state)
        print("top1:%.3f"%prec1)
    else:
        for epoch in range(opt.epochs):
            lr_val = get_lr(optimizer)
            print("Start epoch %d ==========,lr=%f" % (epoch, lr_val))
            train_prec, train_loss = train(state, epoch)
            prec1, val_loss = validate(state)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if not opt.notest or epoch == opt.epochs-1:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict()
                }, is_best, opt.checkpoint_path)
            sw.add_scalars("Accurancy", {'train': train_prec, 'val': prec1}, epoch)
            sw.add_scalars("Loss", {'train': train_loss, 'val': val_loss}, epoch)
            if opt.scheduler == 'plateau':
                scheduler.step(val_loss)
            if opt.scheduler == 'step':
                scheduler.step()
            
            print("top1:%.3f,\tbest_top1:%.3f"%(prec1,best_prec1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action="store_false",default=True)
    parser.add_argument('--notest',action="store_true",default=False)

    parser.add_argument('--dataset', metavar='DIR',
                        default='bird', help='name of the dataset')
    parser.add_argument('--image-size', '-i', default=512, type=int,
                        metavar='N', help='image size (default: 512)')
    parser.add_argument('--input-size', '-cs', default=448, type=int,
                        metavar='N', help='the input size of the model (default: 448)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--optim', default='sgd', type=str,
                        help='the name of optimizer(adam,sgd)')
    parser.add_argument('--scheduler', default='step', type=str,
                        help='the name of scheduler(step,plateau)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')

    parser.add_argument('--smoothing', default=0.1, type=float,
                        metavar='N', help='smoothing')
    parser.add_argument('--model-name', default='resnet', type=str,
                        help='model name')

    parser.add_argument('--use-gpu', action="store_false", default=True,
                        help='whether use gpu or not, default True')
    parser.add_argument('--multi-gpu', action="store_true", default=False,
                        help='whether use multiple gpus or not, default False')
    parser.add_argument('--gpu-ids', default='0',
                        help='gpu id list(eg: 0,1,2...)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint-path', default='checkpoint', type=str, metavar='checkpoint_path',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--loss', default='ce', type=str,
                        help='loss function used for training')

    args = parser.parse_args()
    main(args)
