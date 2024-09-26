import os
import argparse
import torch
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from utils.data.dataloader import create_dataloader
from utils.misc import load_config, build_model, nms
from utils.metrics import Mean, AveragePrecision
import ipdb

class CheckpointManager(object):
    def __init__(self, logdir, model, optim, scaler, scheduler, best_score):
        self.epoch = 0
        self.logdir = logdir
        self.model = model
        self.optim = optim
        self.scaler = scaler
        self.scheduler = scheduler
        self.best_score = best_score

    def save(self, filename):
        data = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_score': self.best_score,
        }
        torch.save(data, os.path.join(self.logdir, filename))

    def restore(self, filename):
        data = torch.load(os.path.join(self.logdir, filename))
        self.model.load_state_dict(data['model_state_dict'])
        self.optim.load_state_dict(data['optim_state_dict'])
        self.scaler.load_state_dict(data['scaler_state_dict'])
        self.scheduler.load_state_dict(data['scheduler_state_dict'])
        self.epoch = data['epoch']
        self.best_score = data['best_score']

    def restore_lastest_checkpoint(self):
        if os.path.exists(os.path.join(self.logdir, 'last.pth')):
            self.restore('last.pth')
            print("Restore the last checkpoint.")

def get_lr(optim):
    for param_group in optim.param_groups:
        return param_group['lr']

def train_step(images, true_boxes, true_classes, model, optim, amp, scaler, metrics, device):
    images = images.to(device)
    true_boxes = [x.to(device) for x in true_boxes]
    true_classes = [x.to(device) for x in true_classes]

    optim.zero_grad()
    with autocast(enabled=amp):
        preds = model(images)
        loss = model.compute_loss(preds, true_boxes, true_classes)
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    loss = loss.item()
    metrics['loss'].update(loss, images.shape[0])

def test_step(images, true_boxes, true_classes, difficulties, model, amp, metrics, device):
    images = images.to(device)
    true_boxes = [x.to(device) for x in true_boxes]
    true_classes = [x.to(device) for x in true_classes]
    difficulties = [x.to(device) for x in difficulties]

    with autocast(enabled=amp):
        preds = model(images)
        loss = model.compute_loss(preds, true_boxes, true_classes)
    loss = loss.item()
    metrics['loss'].update(loss, images.shape[0])

    det_boxes, det_scores, det_classes = nms(*model.decode(preds))
    metrics['APs'].update(det_boxes, det_scores, det_classes, true_boxes, true_classes, difficulties)

def run_grid_search(cfg, args, logdir_suffix, grid_params):
    best_val_loss = float('inf')
    best_params = None
    best_model_state = None
    
    #print("grid_params:", grid_params) 

    for batch_size, lr, momentum, weight_decay in product(
        grid_params['batch_size'], 
        grid_params['optim']['lr'], 
        grid_params['optim']['momentum'], 
        grid_params['optim']['weight_decay']
    ):
        print(f"Training with batch_size={batch_size}, lr={lr}, momentum={momentum}, weight_decay={weight_decay}")

        # Setup model, optimizer, scheduler
        model = build_model(cfg)
        model.to(args.device)
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scaler = GradScaler(enabled=not args.no_amp)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=cfg['training_configs']['scheduler']['milestones'], gamma=cfg['training_configs']['scheduler']['gamma'])

        # Load data
        train_loader = create_dataloader(cfg['train_json'], batch_size=batch_size, image_size=cfg['input_size'], image_mean=cfg['image_mean'], image_stddev=cfg['image_stddev'], augment=True, shuffle=True, num_workers=args.workers)
        val_loader = create_dataloader(cfg['val_json'], batch_size=batch_size, image_size=cfg['input_size'], image_mean=cfg['image_mean'], image_stddev=cfg['image_stddev'], num_workers=args.workers)

        metrics = {'loss': Mean(), 'APs': AveragePrecision(len(cfg['class_names']), cfg['recall_steps'])}
        writers = {'train': SummaryWriter(os.path.join(args.logdir, logdir_suffix, 'train')), 'val': SummaryWriter(os.path.join(args.logdir, logdir_suffix, 'val'))}
                    
        # Train loop
        early_stopping_counter = 0
        for epoch in range(1, cfg['training_configs']['max_epochs'] + 1):
            print(f"Epoch {epoch}/{cfg['training_configs']['max_epochs']}")

            model.train()
            metrics['loss'].reset()
            pbar = tqdm(train_loader, bar_format="{l_bar}{bar:20}{r_bar}", desc="Training")
            for (images, true_boxes, true_classes, _) in pbar:
                train_step(images, true_boxes, true_classes, model, optim, not args.no_amp, scaler, metrics, args.device)
                pbar.set_postfix(loss=f"{metrics['loss'].result:.5f}", lr=get_lr(optim))
                        
            writers['train'].add_scalar('Loss', metrics['loss'].result, epoch)
            writers['train'].add_scalar('Learning rate', get_lr(optim), epoch)
            scheduler.step()

            # Validation
            if epoch % args.val_period == 0:
                model.eval()
                metrics['loss'].reset()
                metrics['APs'].reset()
                with torch.no_grad():
                    for (images, true_boxes, true_classes, difficulties) in tqdm(val_loader, desc="Validation"):
                        test_step(images, true_boxes, true_classes, difficulties, model, not args.no_amp, metrics, args.device)
                            
                val_loss = metrics['loss'].result
                val_pbar.set_postfix(val_loss=f"{val_loss:.5f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stopping_counter = 0
                    best_params = {'batch_size': batch_size, 'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
                    best_model_state = model.state_dict()
                else:
                    early_stopping_counter += 1

                if early_stopping_counter >= cfg['training_configs']['early_stopping']['patience']:
                    print("Early stopping triggered.")
                    break

                writers['val'].add_scalar('Loss', val_loss, epoch)
                writers['val'].add_scalar('mAP@[0.5:0.95]', metrics['APs'].result.mean(), epoch)
                    
        writers['train'].close()
        writers['val'].close()

    return best_val_loss, best_params, best_model_state

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', type=str, required=True, help="config file")
    parser.add_argument('--logdir', type=str, required=True, help="log directory")
    parser.add_argument('--workers', type=int, default=4, help="number of dataloader workers")
    parser.add_argument('--resume', action='store_true', help="resume training")
    parser.add_argument('--no_amp', action='store_true', help="disable automatic mix precision")
    parser.add_argument('--val_period', type=int, default=1, help="number of epochs between successive validation")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    cfg = load_config(args.cfg)

    # Prima Grid Search
    best_val_loss_1, best_params_1, best_model_state_1 = run_grid_search(cfg, args, 'grid_search_1', cfg['training_configs'])

    # Salva i migliori parametri della prima grid search
    torch.save(best_model_state_1, os.path.join(args.logdir, 'best_model_1.pth'))
    print(f"Migliori parametri della prima grid search: {best_params_1}")

    # Seleziona i parametri della seconda grid search
    training_config_2 = cfg['training_config_2']
    lr_index = next(i for i, lst in enumerate(training_config_2['optim']['lr']) if best_params_1['lr'] in lst)
    momentum_index = next(i for i, lst in enumerate(training_config_2['optim']['momentum']) if best_params_1['momentum'] in lst)
    weight_decay_index = next(i for i, lst in enumerate(training_config_2['optim']['weight_decay']) if best_params_1['weight_decay'] in lst)

    second_grid_params = {
        'batch_size': training_config_2['batch_size'],
        'lr': training_config_2['optim']['lr'][lr_index],
        'momentum': training_config_2['optim']['momentum'][momentum_index],
        'weight_decay': training_config_2['optim']['weight_decay'][weight_decay_index],
    }

    # Seconda Grid Search
    best_val_loss_2, best_params_2, best_model_state_2 = run_grid_search(cfg, args, 'grid_search_2', second_grid_params)

    # Salva i migliori parametri della seconda grid search
    torch.save(best_model_state_2, os.path.join(args.logdir, 'best_model_2.pth'))
    print(f"Migliori parametri della seconda grid search: {best_params_2}")

if __name__ == '__main__':
    main()


