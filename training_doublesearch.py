import os
import torch
import yaml
import itertools
import json
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from utils.data.dataloader import create_dataloader
from utils.misc import load_config, build_model, nms
from utils.metrics import Mean, AveragePrecision
import warnings
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
        self.model.load_stateDict(data['model_state_dict'])
        self.optim.loadStateDict(data['optim_state_dict'])
        self.scaler.loadStateDict(data['scaler_state_dict'])
        self.scheduler.loadStateDict(data['scheduler_state_dict'])
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

def run_grid_search(config, args, param_combinations, logdir_prefix):
    # Lista per salvare i risultati
    results = []

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Early stopping parameters
    patience = config['training_configs']['early_stopping']['patience']
    min_delta = config['training_configs']['early_stopping']['min_delta']
    no_improvement_count = 0
    best_score = -float('inf')

    for (batch_size, num_epochs, lr, momentum, weight_decay) in param_combinations:
        print(f"Testing combination: batch_size={batch_size}, epochs={num_epochs}, lr={lr}, momentum={momentum}, weight_decay={weight_decay}")

        # Creare una nuova configurazione
        cfg = config.copy()
        cfg['training_configs']['batch_size'] = batch_size
        cfg['training_configs']['epochs'] = num_epochs
        cfg['training_configs']['optim']['lr'] = lr
        cfg['training_configs']['optim']['momentum'] = momentum
        cfg['training_configs']['optim']['weight_decay'] = weight_decay
        
        # Log directory per questa combinazione
        logdir = os.path.join(args.logdir, f"{logdir_prefix}_bs_{batch_size}_epochs_{num_epochs}_lr_{lr}_momentum_{momentum}_wd_{weight_decay}")
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        # Inizializzare il modello, optimizer, scheduler, e scaler
        model = build_model(cfg)
        model.to(device)
        optim = getattr(torch.optim, cfg['training_configs']['optim']['name'])(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        #scheduler = getattr(torch.optim.lr_scheduler, cfg['training_configs']['scheduler']['name'])(optim, **cfg['training_configs']['scheduler'])
        scheduler_config = cfg['training_configs']['scheduler'].copy()
        scheduler_name = scheduler_config.pop('name')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(optim, **scheduler_config)
        scaler = GradScaler(enabled=not args.no_amp)

        # Checkpointing
        ckpt = CheckpointManager(logdir, model, optim, scaler, scheduler, best_score=0.)
        if args.resume:
            ckpt.restore_lastest_checkpoint()

        # TensorBoard writers
        writers = {
            'train': SummaryWriter(os.path.join(logdir, 'train')),
            'val': SummaryWriter(os.path.join(logdir, 'val'))
        }

        # Dataloader
        train_loader = create_dataloader(cfg['train_json'], batch_size=batch_size, image_size=cfg['input_size'],
                                         image_mean=cfg['image_mean'], image_stddev=cfg['image_stddev'],
                                         augment=True, shuffle=True, num_workers=args.workers, limit=args.limit)
        val_loader = create_dataloader(cfg['val_json'], batch_size=batch_size, image_size=cfg['input_size'],
                                       image_mean=cfg['image_mean'], image_stddev=cfg['image_stddev'],
                                       num_workers=args.workers, limit=args.limit)
        

        # Metrics
        metrics = {
            'loss': Mean(),
            'APs': AveragePrecision(len(cfg['class_names']), cfg['recall_steps'])
        }

        # Training Loop
        for epoch in range(ckpt.epoch + 1, num_epochs + 1):
            model.train()
            metrics['loss'].reset()

            pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}/{num_epochs}", bar_format="{l_bar}{bar:20}{r_bar}")
            for images, true_boxes, true_classes, _ in pbar:
                train_step(images, true_boxes, true_classes, model, optim, not args.no_amp, scaler, metrics, device)
                pbar.set_postfix(loss=metrics['loss'].result, lr=get_lr(optim))

            writers['train'].add_scalar('Loss', metrics['loss'].result, epoch)
            writers['train'].add_scalar('Learning rate', get_lr(optim), epoch)
            scheduler.step()

            # Validation
            if epoch % args.val_period == 0:
                model.eval()
                metrics['loss'].reset()
                metrics['APs'].reset()
                pbar = tqdm(val_loader, desc="Validation", bar_format="{l_bar}{bar:20}{r_bar}")
                with torch.no_grad():
                    for images, true_boxes, true_classes, difficulties in pbar:
                        test_step(images, true_boxes, true_classes, difficulties, model, not args.no_amp, metrics, device)
                        pbar.set_postfix(loss=metrics['loss'].result)

                # Calcolo delle metriche mAP
                APs = metrics['APs'].result
                mAP50 = APs[:, 0].mean()
                mAP = APs.mean()

                if mAP > ckpt.best_score + min_delta:
                    no_improvement_count = 0
                    ckpt.best_score = mAP
                    ckpt.save('best.pth')
                else:
                    no_improvement_count += 1

                writers['val'].add_scalar('Loss', metrics['loss'].result, epoch)
                writers['val'].add_scalar('mAP@[0.5]', mAP50, epoch)
                writers['val'].add_scalar('mAP@[0.5:0.95]', mAP, epoch)
                print(f"mAP@[0.5]: {mAP50:.3f}")
                print(f"mAP@[0.5:0.95]: {mAP:.3f} (best: {ckpt.best_score:.3f})")

                # Early Stopping Check
                if no_improvement_count >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            ckpt.epoch += 1
            ckpt.save('last.pth')

        # Chiudere i writers di TensorBoard
        writers['train'].close()
        writers['val'].close()

        # Salvataggio dei risultati per questa combinazione
        results.append({
            'batch_size': batch_size,
            'epochs': num_epochs,
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'best_score': ckpt.best_score
        })

    return results

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', type=str, required=True, help="config file")
    parser.add_argument('--logdir', type=str, required=True, help="log directory")
    parser.add_argument('--workers', type=int, default=4, help="number of dataloader workers")
    parser.add_argument('--resume', action='store_true', help="resume training")
    parser.add_argument('--no_amp', action='store_true', help="disable automatic mix precision")
    parser.add_argument('--val_period', type=int, default=1, help="number of epochs between successive validation")
    parser.add_argument('--limit', type=int, default=None, help='limit the number of images for training and validation')
    args = parser.parse_args()

    config = load_config(args.cfg)
    ipdb.set_trace()
    
    
    # First Grid Search
    batch_sizes = config['training_configs']['batch_size']
    epochs_list = config['training_configs']['epochs']
    learning_rates = config['training_configs']['optim']['lr']
    momentums = config['training_configs']['optim']['momentum']
    weight_decays = config['training_configs']['optim']['weight_decay']
    param_combinations = list(itertools.product(batch_sizes, epochs_list, learning_rates, momentums, weight_decays))
    results = run_grid_search(config, args, param_combinations, "first_grid")

    # Find the best parameters from the first grid search
    best_params = max(results, key=lambda x: x['best_score'])
    best_lr = best_params['lr']
    best_momentum = best_params['momentum']
    best_weight_decay = best_params['weight_decay']

    # Second Grid Search with training_config_2
    new_lrs = config['training_config_2']['optim']['lr']
    new_momentums = config['training_config_2']['optim']['momentum']
    new_weight_decays = config['training_config_2']['optim']['weight_decay']
    new_param_combinations = list(itertools.product(batch_sizes, epochs_list, new_lrs, new_momentums, new_weight_decays))
    new_results = run_grid_search(config, args, new_param_combinations, "second_grid")

    # Save both results
    all_results = {
        'first_grid_search': results,
        'second_grid_search': new_results
    }
    results_json_file = os.path.join(args.logdir, 'all_grid_search_results.json')
    with open(results_json_file, 'w') as file:
        json.dump(all_results, file, indent=4)

if __name__ == '__main__':
    main()


