import torch
import numpy as np
from utils.boxes import calculate_ious
from utils.constants import BACKGROUND_INDEX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mean(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, v, n):
        self.sum += v
        self.count += n

    @property
    def result(self):
        return self.sum / self.count


class AveragePrecision(object):
    def __init__(self, num_classes, recall_steps):
        self.num_classes = num_classes
        self.recall_steps = recall_steps
        self.reset()

    def reset(self):
        self.num_relevent = torch.zeros([self.num_classes], device=device)
        self.det_scores = []
        self.det_classes = []
        self.true_positives = []
        self.false_positives = []

    def update(self, det_boxes, det_scores, det_classes, true_boxes, true_classes,
               difficulties):
        iou_thres = np.linspace(0.5, 0.95, 10)

        bs = len(det_boxes)
        for i in range(bs):
            num_dets = det_boxes[i].shape[0]
            num_true = true_boxes[i].shape[0]
            if num_true == 0:
                self.true_positives.extend([False] * num_dets)
                self.false_positives.extend([True] * num_dets)
                continue

            # Count relevent elements
            class_indices, counts = torch.unique(
                true_classes[i][
                    (true_classes[i] != BACKGROUND_INDEX)
                    & (difficulties[i] == 0)
                ],
                return_counts=True
            )
            #self.num_relevent[class_indices] += counts.cpu()
            self.num_relevent[class_indices.to(device)] += counts.to(device) 

            if num_dets == 0:
                continue

            # Store detections
            self.det_scores.extend(det_scores[i].tolist())
            self.det_classes.extend(det_classes[i].tolist())

            # Determine if detections match ground truth in terms of IoU and class label
            #ious = calculate_ious(true_boxes[i], det_boxes[i])   # [num_true, num_dets]
            ious = calculate_ious(true_boxes[i], det_boxes[i]).to(device)
            """mask = (   # [num_true, num_dets]
                (ious >= iou_thres[0])
                & (true_classes[i].unsqueeze(-1) == det_classes[i])
                & (true_classes[i].unsqueeze(-1) != BACKGROUND_INDEX)
            )"""
            mask = (  # <== Modificato
                (ious >= iou_thres[0])
                & (true_classes[i].unsqueeze(-1).to(device) == det_classes[i].to(device))  # <== Modificato
                & (true_classes[i].unsqueeze(-1).to(device) != BACKGROUND_INDEX)  # <== Modificato
            )

            is_tp = np.full([num_dets, 10], False)
            is_fp = np.full([num_dets, 10], True)
            indices = torch.where(mask)   # ground truth indices, detection indices
            if indices[0].shape[0] > 0:
                matches = torch.stack(indices, axis=1)   # [num_matches, 2]
                matches = matches.cpu().numpy()   # only np.unique() supports `return_index` option
                ious = ious.cpu().numpy()
                ious = ious[matches[:, 0], matches[:, 1]]

                # Find the ground truth with the best IoU for each detection
                matches = matches[ious.argsort()[::-1]]
                indices = np.unique(matches[:, 1], return_index=True)[1]
                matches = matches[indices]
                ious = ious[indices]

                is_difficult = difficulties[i].bool().cpu().numpy()[matches[:, 0]]
                is_fp[matches[:, 1][is_difficult]] = (
                    np.expand_dims(ious[is_difficult], axis=-1)
                    < iou_thres
                )

                # Each ground truth is assigned to the detection with the best IoU
                indices = np.unique(matches[:, 0], return_index=True)[1]
                matches = matches[indices]
                ious = ious[indices]

                is_easy = ~(difficulties[i].bool().cpu().numpy()[matches[:, 0]])
                is_tp[matches[:, 1][is_easy]] = (
                    np.expand_dims(ious[is_easy], axis=-1)
                    >= iou_thres
                )
                is_fp[is_tp] = False

            self.true_positives.extend(is_tp.tolist())
            self.false_positives.extend(is_fp.tolist())

    @property
    def result(self):
        """
        Returns:
            APs: float32 tensor. Shape: [num_classes, 10].
        """
        scores, indices = torch.sort(
            torch.FloatTensor(self.det_scores).to(device),
            descending=True
            )
        #classes = torch.IntTensor(self.det_classes)[indices]
        classes = torch.IntTensor(self.det_classes).to(device)[indices]
        #true_positives = torch.IntTensor(self.true_positives)[indices]
        true_positives = torch.IntTensor(self.true_positives).to(device)[indices]
        #false_positives = torch.IntTensor(self.false_positives)[indices]
        false_positives = torch.IntTensor(self.false_positives).to(device)[indices]
        #recall_thres = torch.linspace(0, 1, self.recall_steps)
        recall_thres = torch.linspace(0, 1, self.recall_steps).to(device)
        
        #APs = torch.zeros([self.num_classes, 10])
        APs = torch.zeros([self.num_classes, 10], device=device) 
        for c in range(self.num_classes):
            if (classes == c).int().sum() == 0:
                continue
            
            #tp_cum = torch.cumsum(true_positives[classes == c].to(device), axis=0)  # [n, 10]
            tp_cum = torch.cumsum(true_positives[classes == c].to(device), axis=0) 
            #fp_cum = torch.cumsum(false_positives[classes == c].to(device), axis=0) # [n, 10]
            fp_cum = torch.cumsum(false_positives[classes == c].to(device), axis=0)
            num_relevent = self.num_relevent[c]

            recalls = tp_cum / num_relevent
            precisions = tp_cum / (tp_cum + fp_cum + 1e-10)

            recall_above_thres = recalls.unsqueeze(-1) >= recall_thres   # [n, 10, recall_steps]

            max_precisions, _ = torch.max(   # [10, recall_steps]
                precisions.unsqueeze(-1) * recall_above_thres.int(),
                axis=0
            )
            APs[c] = torch.mean(max_precisions, axis=-1)
        return APs
