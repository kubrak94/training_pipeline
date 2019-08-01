from collections import OrderedDict
from copy import copy
import io
import json
from os.path import join, exists, isfile
from os import makedirs
import re
import shutil
import time
from typing import NamedTuple, List

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import yaml

import src.data as data
import src.models as models
from src.train import learning_rate
from src.train import loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LossInfo(NamedTuple):
    name: str
    loss: torch.nn.Module
    alpha: int = 1


class MetricInfo(NamedTuple):
    name: str
    values: List[float]
        
        
def fix_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        

def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    # so we can treat 5e-6 as a number, not a string
    # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    OrderedLoader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    return yaml.load(stream, OrderedLoader)


def load_hparams(hparams_path):
    if not exists(hparams_path):
        raise Exception('You must provide path to existing .yaml file with hyperparameters')

    with open(hparams_path, 'rb') as f:
        hparams = ordered_load(f)

    return hparams


def create_logdir(hparams):
    """
    Create log directory: f'/home/konstantin/training_pipeline/models/{model_name}_{in_channels}_{out_channels}_{start_time}'
    """
    model_params = hparams['model']
    model_name = model_params['name']
    
    # by default assume that we have 1 input and 1 output channel
    model_in_channels = model_params['in_channels'] if 'in_channels' in model_params else 1
    model_out_channels = model_params['out_channels'] if 'out_channels' in model_params else 1
    cur_time = time.ctime().replace(' ', '_').replace(':', '-')
    
    model_dir = f'/home/konstantin/training_pipeline/models/{model_name}_{model_in_channels}_{model_out_channels}_{cur_time}'

    if not exists(model_dir):
        makedirs(model_dir)

    return model_dir


def prepare_model(hparams):
    if 'model' not in hparams:
        raise Exception('You must add model params to hyperparams')

    model_params = hparams['model']
    model_name = model_params.pop('name')

    model = models.__dict__[model_name](**model_params)
    model.apply(models.weight_init)
    model.to(device)
    
    return model


def create_total_loss(loss_functions):
    return lambda x, y: sum(lf.alpha * lf.loss(x, y) for lf in loss_functions)


def prepare_training(hparams, model):
    """Converts given hyperparameters to instances of classes.
       Arguments:
           hparams: dict, hyperparameters
           model: torch.nn.Model, instanse of model to be trained
       Returns:
           loss_functions – torch.nn.Module, loss functions to be evaluated 
           optimizer – torch.optim, optimizer for the model
           scheduler – torhc.nn.lr_scheduler, learning rate scheduler
    """
    if 'losses' not in hparams:
        raise Exception('You must add loss params to hparams')

    losses_params = hparams['losses']
    loss_functions = []
    metrics = []
    
    for loss_params in losses_params:
        loss_name = loss_params.pop('name')
        
        if 'alpha' in loss_params:
            loss_alpha = loss_params.pop('alpha')
        else:
            loss_alpha = 1

        if loss_name not in ['MSELoss', 'L1Loss', 'MultiLabelMarginLoss', 'MSSSIM']:
            weights_path = loss_params.pop('weights_path')

            with open(weights_path, 'r') as infile:
                loss_params["weight"] = torch.Tensor(json.load(infile))

        loss_function = loss.__dict__[loss_name](**loss_params)
        loss_function = loss_function.to(device)
        loss_functions.append(LossInfo(loss_name, loss_function, loss_alpha))
        
    total_loss = loss.TotalLoss(loss_functions)

    if 'optimizer' not in hparams:
        raise Exception('You must add optimizer params to hparams')

    optimizer_params = hparams['optimizer']
    optimizer_name = optimizer_params.pop('name')
    optimizer = torch.optim.__dict__[optimizer_name](
        filter(lambda p: p.requires_grad, model.parameters()),
        **optimizer_params
    )

    if 'scheduler' in hparams:
        scheduler_params = hparams['scheduler']
        if 'name' not in scheduler_params:
            raise Exception('If you provided scheduler params you also must add scheduler name')
        scheduler_name = scheduler_params.pop('name')

        scheduler = learning_rate.__dict__[scheduler_name](
            optimizer, **scheduler_params
        )
    else:
        scheduler = None

    return total_loss, loss_functions, optimizer, scheduler


def prepare_dataloader(hparams, mode):
    assert mode in ['train', 'test', 'validation'], "The mode should be one of: 'train', 'test', 'validation'."
    
    dataloader_params = copy(hparams['dataset'])
    dataloader_name = dataloader_params.pop('name')
    
    augmentations = dataloader_params.pop('augmentations')
    train_csv_file = dataloader_params.pop('train_csv_file')
    val_csv_file = dataloader_params.pop('val_csv_file')
    test_csv_file = dataloader_params.pop('test_csv_file')
    
    if mode == 'train':
        transforms = data.prepare_transforms(augmentations['train'])
        csv_file = train_csv_file
    else:
        transforms = data.prepare_transforms(augmentations['validation'])
        dataloader_params.pop('repeat_dataset')
        if mode == 'test':
            csv_file = test_csv_file
        else:
            csv_file = val_csv_file
    
    dataloader_params['transforms'] = transforms
    dataloader_params['csv_file'] = csv_file
    
    dataset = data.__dict__[dataloader_name](**dataloader_params)
    
    batch_size = hparams['batch_size']
    num_workers = hparams['num_workers']
    shuffle = mode == 'train'
    
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())    


def add_metrics(summary_writer, epoch_metrics, epoch, mode):
    """
    Adds metrics to Tensorboard
    :param summary_writer: SummaryWriter instance
    :param epoch_metrics: dict with epoch metrics
    :param epoch: number of epoch
    :param mode: train (for training) or val (for validation)
    :return: None
    """
    for metric_name, metric_value in epoch_metrics.items():
        m_value = metric_value
        if metric_name in ['inference_result']:
            m_value = prepare_infer_image(m_value)
            summary_writer.add_image('{}/{}'.format(mode, metric_name), m_value, epoch + 1)
        else:
            summary_writer.add_scalar('{}/{}'.format(mode, metric_name), m_value, epoch + 1)


def prepare_infer_image(infer_result):
    """
    Transforms output of neural network to image that can be shown at tensorboard
    :param infer_result: torch.Tensor, shape (C, W, H) - channels, width, height, representing 
                         output from the neural network
    :return: normalized output from the neural network moved to cpu()
    """
    denominator = torch.max(infer_result[0]) - torch.min(infer_result[0])
    image = (infer_result[0].cpu() - torch.min(infer_result)) / denominator

    return image.unsqueeze_(0)


def run_train_val_loader(epoch, loader, mode, model, 
                         loss_functions, total_loss, optimizer, scheduler, train_iter=1):
    """
    Runs one epoch of the training loop.
    :param epoch: index of the epoch
    :param loader: data loader
    :param mode: 'train' or 'valid'
    :param model: model for training \ validation
    :param loss_functions: loss function to minimize
    :param optimizer: optimisation function
    :param train_iter: how many times iterate through train dataset
    :return: None
    """
    if mode == 'train':
        model.train()
    else:
        model.eval()
        
    repeat_num = train_iter if mode == 'train' else 1
    
    epoch_metrics = {l.name: [] for l in loss_functions}
    epoch_metrics['total_loss'] = []

    for _ in range(0, repeat_num):
        for i, batch in tqdm(enumerate(loader), total=len(loader)):
            inputs = batch['input']
            targets = batch['target']

            inputs, targets = inputs.to(device), targets.to(device)

            with torch.set_grad_enabled(mode == 'train'):
                output = model.forward(inputs)
                    
                for criterion in loss_functions:
                    value = criterion.loss(output, targets)
                    epoch_metrics[criterion.name].append(value.data.cpu().numpy().astype(np.float))

                #total_loss_ = total_loss(output, targets)
                #epoch_metrics['total_loss'].append(total_loss_.data.cpu().numpy().astype(np.float))
                
                # TODO: remake this alpha scheduler to work internally
                if epoch == 0:
                    total_loss_ = loss.MSELoss()(output, targets)
                elif 0.5 < loss_functions[0].alpha < 1 and len(loss_functions) == 2:
                    alpha = alpha_scheduler(epoch, loss_functions[0].alpha)
                    total_loss_ = alpha * loss_functions[0].loss(output, targets) + (1 - alpha) * loss_functions[1].loss(output, targets)
                elif 0 < loss_functions[0].alpha < 0.5 and len(loss_functions) == 2:
                    alpha = alpha_scheduler(epoch, loss_functions[1].alpha)
                    total_loss_ = (1 - alpha) * loss_functions[0].loss(output, targets) + alpha * loss_functions[1].loss(output, targets)
                else:
                    total_loss_ = total_loss(output, targets)
                epoch_metrics['total_loss'].append(total_loss_.data.cpu().numpy().astype(np.float))

                if mode == 'train':
                    optimizer.zero_grad()
                    total_loss_.backward()
                    optimizer.step()

            if isinstance(scheduler, learning_rate.CyclicLR):
                scheduler.batch_step()
    
    for metric_name, metric_values in epoch_metrics.items():
        epoch_metrics[metric_name] = np.mean(np.array(metric_values))
    
    print("{epoch} * Epoch ({mode}): ".format(epoch=epoch, mode=mode), epoch_metrics['total_loss'])
    epoch_metrics['inference_result'] = output[-1]

    return epoch_metrics


def save_checkpoint(model, optimizer, epoch, best_metrics, logdir):
    state = {
        "epoch": epoch,
        "best_metrics": best_metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }

    if not exists(logdir):
        makedirs(logdir)

    filename = "{}/checkpoint.pth.tar".format(logdir)
    torch.save(state, filename)


def load_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    start_epoch = checkpoint['epoch'] + 1
    best_metrics = checkpoint['best_metrics']
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    hparams = checkpoint['hparams']

    return start_epoch, best_metrics, \
           model_state_dict, optimizer_state_dict, hparams




    # TODO: move to a separate module
def alpha_scheduler(epoch, base=0.999):

    def magic_function(epoch, multiplier=2):
        a = (epoch - 15) / multiplier
        return (np.exp(a) / (np.exp(a) + 1) )

    return base * magic_function(epoch)




def run_train(args):
    hparams = load_hparams(args.hparams)

    seed = hparams['seed']
    fix_random_seed(seed)

    model_dir = create_logdir(hparams)
    with open(join(model_dir, 'config.yaml'), 'w') as outfile:
        yaml.dump(hparams, outfile, default_flow_style=False)

    model = prepare_model(hparams)
    total_loss, loss_functions, optimizer, scheduler = prepare_training(hparams, model)
    
    train_loader = prepare_dataloader(hparams, 'train')
    valid_loader = prepare_dataloader(hparams, 'validation')
    
    train_iter = hparams['train_iter']
    
    summary_writer = SummaryWriter(model_dir)
    best_metrics = np.inf

    start_epoch = 0

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            start_epoch, best_metrics, \
            model_state_dict, optimizer_state_dict, hparams = load_checkpoint(args.resume)

            new_state_dict = OrderedDict()
            for k, v in model_state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            optimizer.load_state_dict(optimizer_state_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, start_epoch - 1))
        else:
            raise Exception("no checkpoint found at '{}'".format(args.resume))

    print('Training started')
    epochs = hparams['epochs']

    for epoch in range(start_epoch, epochs):
        is_best = False
        train_metrics = run_train_val_loader(epoch, train_loader, 'train', model,
                                             loss_functions, total_loss, optimizer, scheduler, train_iter)
        valid_metrics = run_train_val_loader(epoch, valid_loader, 'valid', model,
                                             loss_functions, total_loss, optimizer, scheduler)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_loss)
        elif (scheduler is not None) and (not isinstance(scheduler, learning_rate.CyclicLR)):
            scheduler.step()

        if valid_metrics['total_loss'] < best_metrics:
            best_metrics = valid_metrics['total_loss']
            is_best = True

        if is_best:
            save_checkpoint(model, optimizer, epoch,
                            best_metrics, logdir=model_dir)

        add_metrics(summary_writer, train_metrics, epoch, 'train')
        add_metrics(summary_writer, valid_metrics, epoch, 'valid')

