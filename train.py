import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn

from config import device, num_classes, grad_clip, print_freq
from data_gen import MICDataset
from models.deeplab import DeepLab
from utils import parse_args, save_checkpoint, AverageMeter, clip_gradient, get_logger, get_learning_rate, accuracy


# from torch.optim.lr_scheduler import MultiStepLR


def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        model = DeepLab(backbone='mobilenet', output_stride=16, num_classes=num_classes)
        model = nn.DataParallel(model)

        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom, nesterov=True)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)

    # Custom dataloaders
    train_dataset = MICDataset('train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    valid_dataset = MICDataset('valid')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    # Epochs
    for epoch in range(start_epoch, args.end_epoch):
        # One epoch's training
        train_loss, train_acc = train(train_loader=train_loader,
                                      model=model,
                                      optimizer=optimizer,
                                      epoch=epoch,
                                      logger=logger)
        effective_lr = get_learning_rate(optimizer)
        print('Current effective learning rate: {}\n'.format(effective_lr))

        writer.add_scalar('model/train_loss', train_loss, epoch)
        writer.add_scalar('model/train_acc', train_acc, epoch)

        # One epoch's validation
        valid_loss, valid_acc = valid(valid_loader=valid_loader,
                                      model=model,
                                      logger=logger)

        writer.add_scalar('model/valid_loss', valid_loss, epoch)
        writer.add_scalar('model/valid_acc', valid_acc, epoch)

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)
        # scheduler.step(epoch)


def train(train_loader, model, optimizer, epoch, logger):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    accs = AverageMeter()

    # Batches
    for i, (img, y) in enumerate(train_loader):
        # Move to GPU, if available
        img = img.float().to(device)  # [N, 1, 256, 256]
        y = y.to(device)  # [N, 313, 64, 64]

        # Forward prop.
        y_hat = model(img)  # [N, 313, 64, 64]

        # Calculate loss
        # loss = criterion(out, target)
        loss = -y * (1 - y_hat).pow(2) * torch.log(y_hat)  # [N, 313, 64, 64]
        loss = torch.sum(loss, dim=1)  # [N, 64, 64]
        loss = loss.mean()
        acc = accuracy(y_hat, y)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        clip_gradient(optimizer, grad_clip)

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        accs.update(acc)

        # Print status
        if i % print_freq == 0:
            status = 'Epoch: [{0}][{1}/{2}]\t' \
                     'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                     'Accuracy {acc.val:.5f} ({acc.avg:.5f})\t'.format(epoch, i, len(train_loader), loss=losses,
                                                                       acc=accs)
            logger.info(status)

    return losses.avg, accs.avg


def valid(valid_loader, model, logger):
    model.eval()  # eval mode (dropout and batchnorm is NOT used)

    losses = AverageMeter()
    accs = AverageMeter()

    # Batches
    for img, y in valid_loader:
        # Move to GPU, if available
        img = img.float().to(device)  # [N, 1, 256, 256]
        y = y.to(device)  # [N, 313, 64, 64]

        # Forward prop.
        y_hat = model(img)  # [N, 313, 64, 64]

        # Calculate loss
        # loss = criterion(out, target)
        loss = -y * (1 - y_hat).pow(2) * torch.log(y_hat)  # [N, 313, 64, 64]
        loss = torch.sum(loss, dim=1)  # [N, 64, 64]
        loss = loss.mean()
        acc = accuracy(y_hat, y)

        # Keep track of metrics
        losses.update(loss.item())
        accs.update(acc)

    # Print status
    status = 'Validation: Loss {loss.avg:.5f}\t Accuracy {acc.avg:.5f}\n'.format(loss=losses, acc=accs)
    logger.info(status)

    return losses.avg, accs.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
