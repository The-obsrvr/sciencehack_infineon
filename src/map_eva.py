import torch
from collection import Counter
from iou import intersection_over_union

def mean_avg_precision(
    pred_boxes, true_boxes, iou_threshold = 0.5, box_format="corners", num_classes = 5):
    # pred_boxes (list):[[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]
    avg_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # amount_boxes = {0:torch.tensor([0,0,0}), 1:torch.tensor([0,0,0,0,0])}
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format = box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
                
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            
            # for calculating precision and recall
            # [1, 1, 0, 1, o] -> [1, 2, 2, 3, 3]
            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
            precisions = torch.cat((torch.tensor([1]), precisions))
            recalls = torch.cat((torch.tensor([0]), recalls))

            avg_precisions.append(torch.trapz(precisions, recalls))

        return sum(avg_precisions) / len(avg_precisions)