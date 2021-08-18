import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

from collections import defaultdict
from utils.data.load_data import create_train_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.unet import Unet
from utils.model.mnet import Mnet
from utils.model.munet import MUnet

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        input, target, maximum, _, _ = data
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(input)
        loss = loss_type(output, target, maximum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            duration = time.perf_counter() - start_iter
            remain_iter = len(data_loader) - iter
            remain_time = remain_iter * duration
            minutes = int(remain_time // 60)
            seconds = int(remain_time % 60)
    
            
            print(
                f'Epoch = [{epoch+1:3d}/{args.num_epochs:3d}] '+
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '+
                f'Loss = {loss.item():.4f} '+
                f'Time = {duration:.4f}s '+
                f'ETA = {minutes:2d}m {seconds:2d}s '
            )
            
        start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader, loss_type):
    len_loader = len(data_loader)
    total_loss = 0

    start = time.perf_counter()
    
    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, maximum, _, _ = data
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)

            output = model(input)
            loss = loss_type(output, target, maximum)
            total_loss += loss.item()

    total_loss = total_loss / len_loader

    return total_loss, time.perf_counter() - start

    '''
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    inputs = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, _, fnames, slices = data
            input = input.cuda(non_blocking=True)
            output = model(input)

            for i in range(output.shape[0]):
                fname = fnames[i]
                int_slice = int(slices[i])
                reconstructions[fname][int_slice] = output[i].cpu().numpy()
                targets[fname][int_slice] = target[i].numpy()
                inputs[fname][int_slice] = input[i].cpu().numpy()


    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    for fname in inputs:
        inputs[fname] = np.stack(
            [out for _, out in sorted(inputs[fname].items())]
        )
        metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, inputs, time.perf_counter() - start
    '''


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def early_stopping(val_loss_data, epoch, args):
    min_delta = args.min_delta
    patient = args.patient

    if epoch <= patient:
      return False
    
    for i in range(patient, 0, -1):
      if val_loss_data[-i-1] - val_loss_data[-i] > min_delta:
        return False
    
    return True



        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    
    if str(args.net_name) == 'Unet':
      model = Unet(in_chans = args.in_chans, out_chans = args.out_chans)
    elif str(args.net_name) == 'Mnet':
      model = Mnet(in_chans = args.in_chans, out_chans = args.out_chans)
    elif str(args.net_name) == 'MUnet':
      model = MUnet(in_chans = args.in_chans, out_chans = args.out_chans)
    else:
      raise Exception(f'Unknown network: {args.net_name}')

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)

    model.to(device=device)
    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    best_val_loss = 1.
    start_epoch = 0

    train_loss_data = []
    val_loss_data = []

    train_loader, val_loader = create_train_data_loaders(data_path = args.data_path_train, args = args)


    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch+1:2d} ............... {args.net_name} ...............')

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        
        '''
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        
        val_loss = val_loss / num_subjects
        '''
        val_loss, val_time = validate(args, model, val_loader, loss_type)

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        train_loss_data.append(train_loss)
        val_loss_data.append(val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        
        print(
            f'Epoch = [{epoch+1:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            '''
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
            '''

        if early_stopping(val_loss_data, epoch, args):
          print(f'Early Stopping. Epochs: {epoch+1}')
          break

    
    plt.plot(train_loss_data, label='train')
    plt.plot(val_loss_data, label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png', dpi=300)