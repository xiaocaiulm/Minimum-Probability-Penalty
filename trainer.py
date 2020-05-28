from utils import accuracy, AverageMeter
import time
import torch
from tqdm import tqdm


def train(state, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model = state['model']
    config = state['config']
    criterion = state['criterion']
    optimizer = state['optimizer']

    # switch to train mode
    model.train()

    end = time.time()
    print(('\n' + '%10s' * 4) % ('Epoch', 'gpu_mem', 'loss', 'top1'))
    pbar = tqdm(enumerate(state['train_loader']), total=len(state['train_loader']))
    for i, (input, target) in pbar:
        data_time.update(time.time() - end)
        use_gpu = torch.cuda.is_available() and config.use_gpu
        if use_gpu:
            target = target.cuda()
            input = input.cuda()

        output = model(input)
        loss = criterion(output, target)

        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if use_gpu else 0)
        s = ('%10s' * 2 + '%10.3g' * 2) % ('%g/%g' % (epoch, config.epochs - 1), mem, losses.avg, top1.avg)
        pbar.set_description(s)
    return top1.avg, losses.avg


def validate(state):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model = state['model']
    config = state['config']
    criterion = state['criterion']

    model.eval()

    end = time.time()
    print(('\n' + '%10s' * 3) % ('gpu_mem', 'loss', 'top1'))
    pbar = tqdm(enumerate(state['val_loader']), total=len(state['val_loader']))
    for i, (input, target) in pbar:
        use_gpu = torch.cuda.is_available() and config.use_gpu
        if use_gpu:
            target = target.cuda()
            input = input.cuda()
        with torch.no_grad():
            pre = model(input)
            loss = criterion(pre, target)
            losses.update(loss.item(), input.size(0))

            prec1 = accuracy(pre.data, target, topk=(1,))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if use_gpu else 0)
            s = ('%10s' * 1 + '%10.3g' * 2) % (mem, losses.avg, top1.avg)
            pbar.set_description(s)

    return top1.avg, losses.avg
