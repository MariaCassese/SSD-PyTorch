import torch
import argparse
import os
import json
import tempfile
import csv
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard writer
from utils.boxes import xyxy2xywh
from utils.misc import load_config, build_model, nms

def main():
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', type=str, required=True,
                        help="config file")
    parser.add_argument('--coco_dir', type=str, required=True,
                        help="path to a directory containing COCO 2017 dataset.")
    parser.add_argument('--pth', type=str, required=True,
                        help="checkpoint")
    parser.add_argument('--no_amp', action='store_true',
                        help="disable automatic mix precision")
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help="TensorBoard log directory")
    parser.add_argument('--output_image', type=str, default='histogram.png',
                        help="Path to save the histogram image")
    parser.add_argument('--times_file', type=str, default='processing_times.csv',
                        help="File to save processing times")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = load_config(args.cfg)

    model = build_model(cfg)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(args.pth)['model_state_dict'])

    preprocessing = Compose(
        [
            Resize((cfg.input_size,) * 2),
            ToTensor(),
            Normalize([x / 255 for x in cfg.image_mean], [x / 255 for x in cfg.image_stddev]),
        ]
    )

    coco = COCO(os.path.join(args.coco_dir, 'annotations/instances_val2017.json'))
    cat_ids = coco.getCatIds()
    results = []
    processing_times = []

    writer = SummaryWriter(log_dir=args.log_dir)  # Initialize TensorBoard writer

    with torch.no_grad():
        for k, v in tqdm(coco.imgs.items()):
            
            # starting time of the process
            start_time = timer()

            image_path = os.path.join(args.coco_dir, 'val2017/%s' % v['file_name'])
            image = Image.open(image_path).convert('RGB')
            image = preprocessing(image)
            image = image.unsqueeze(0).to(device)

            with autocast(enabled=(not args.no_amp)):
                preds = model(image)
            det_boxes, det_scores, det_classes = nms(*model.decode(preds))
            det_boxes, det_scores, det_classes = det_boxes[0], det_scores[0], det_classes[0]

            det_boxes = torch.clip(det_boxes / cfg.input_size, 0, 1)
            det_boxes = (
                det_boxes.cpu()
                * torch.FloatTensor([v['width'], v['height']]).repeat([2])
            )
            det_boxes = xyxy2xywh(det_boxes)

            det_boxes, det_scores, det_classes = (
                det_boxes.tolist(),
                det_scores.tolist(),
                det_classes.tolist(),
            )

            det_classes = [cat_ids[c] for c in det_classes]

            for box, score, clss in zip(det_boxes, det_scores, det_classes):
                results.append(
                    {
                        'image_id': k,
                        'category_id': clss,
                        'bbox': box,
                        'score': score
                    }
                )
            #end time
            end_time = timer()
            processing_times.append(end_time - start_time)

    _, tmp_json = tempfile.mkstemp('.json')
    with open(tmp_json, 'w') as f:
        json.dump(results, f)
    results = coco.loadRes(tmp_json)
    coco_eval = COCOeval(coco, results, 'bbox')
    coco_eval.params.imgIds = list(coco.imgs.keys())   
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Log the evaluation metrics in TensorBoard
    metrics = {
        "AP": coco_eval.stats[0],
        "AP@.5": coco_eval.stats[1],
        "AP@.75": coco_eval.stats[2],
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
        "AR@1": coco_eval.stats[6],
        "AR@10": coco_eval.stats[7],
        "AR@100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11]
    }
    for metric, value in metrics.items():
        writer.add_scalar(metric, value)
    
    writer.close()  # Close the writer
    
    # Ottieni il minimo e il massimo dei tempi di elaborazione
    min_time = min(processing_times)
    max_time = max(processing_times)
    print(f"Tempo minimo per immagine: {min_time:.4f} secondi")
    print(f"Tempo massimo per immagine: {max_time:.4f} secondi")
    
    # Salva i tempi di elaborazione in un file CSV
    with open(args.times_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Tempo per immagine (secondi)"])  # Intestazione
        for time in processing_times:
            writer.writerow([time])  # Salva ciascun tempo in una nuova riga
    print(f"Tempi di elaborazione salvati in {args.times_file}")

    # Crea e salva un istogramma dei tempi di elaborazione limitato tra min e max
    """plt.hist(processing_times, bins=4, range=(min_time, max_time))
    plt.xlabel("Tempo per immagine (secondi)")
    plt.ylabel("Frequenza")
    plt.title("Distribuzione dei tempi di elaborazione per immagine")
    plt.savefig(args.output_image)  # Salva l'istogramma come immagine
    print(f"Istogramma salvato in {args.output_image}")"""

if __name__ == '__main__':
    main()
    
