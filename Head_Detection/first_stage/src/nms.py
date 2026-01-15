# src/nms.py
from __future__ import annotations
import torch

def _box_area_xyxy(b: torch.Tensor) -> torch.Tensor:
    return (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

def box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    a1 = _box_area_xyxy(box1)
    a2 = _box_area_xyxy(box2)

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = a1[:, None] + a2[None, :] - inter + 1e-9
    return inter / union

def box_diou_xyxy(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    iou = box_iou_xyxy(box1, box2)

    b1_cx = (box1[:, 0] + box1[:, 2]) * 0.5
    b1_cy = (box1[:, 1] + box1[:, 3]) * 0.5
    b2_cx = (box2[:, 0] + box2[:, 2]) * 0.5
    b2_cy = (box2[:, 1] + box2[:, 3]) * 0.5
    rho2 = (b1_cx[:, None] - b2_cx[None, :])**2 + (b1_cy[:, None] - b2_cy[None, :])**2

    en_x1 = torch.min(box1[:, None, 0], box2[None, :, 0])
    en_y1 = torch.min(box1[:, None, 1], box2[None, :, 1])
    en_x2 = torch.max(box1[:, None, 2], box2[None, :, 2])
    en_y2 = torch.max(box1[:, None, 3], box2[None, :, 3])
    c2 = (en_x2 - en_x1)**2 + (en_y2 - en_y1)**2 + 1e-9

    return iou - (rho2 / c2)


def standard_nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        iou = box_iou_xyxy(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
        order = rest[iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long)

def diou_nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, diou_thr: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        diou = box_diou_xyxy(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
        order = rest[diou <= diou_thr]
    return torch.tensor(keep, dtype=torch.long)

def soft_nms_xyxy(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_thr: float = 0.5,
    sigma: float = 0.5,
    score_thr: float = 0.001,
    method: str = "gaussian",
) -> torch.Tensor:

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    method = method.lower()
    assert method in ("gaussian", "linear")

    idxs = torch.arange(boxes.size(0), device=boxes.device)
    b = boxes.clone()
    s = scores.clone()

    keep = []
    while idxs.numel() > 0:
        m = torch.argmax(s)
        i = idxs[m].item()
        keep.append(i)

        b0 = b[m].unsqueeze(0)
        s0 = s[m].clone()

        b = torch.cat([b[:m], b[m+1:]], dim=0)
        s = torch.cat([s[:m], s[m+1:]], dim=0)
        idxs = torch.cat([idxs[:m], idxs[m+1:]], dim=0)

        if idxs.numel() == 0:
            break

        iou = box_iou_xyxy(b0, b).squeeze(0)

        if method == "linear":
            decay = torch.ones_like(iou)
            mask = iou > iou_thr
            decay[mask] = 1.0 - iou[mask]
        else:
            # gaussian
            decay = torch.exp(-(iou * iou) / sigma)

        s = s * decay
        # drop low-score
        keep_mask = s >= score_thr
        b = b[keep_mask]
        s = s[keep_mask]
        idxs = idxs[keep_mask]

    return torch.tensor(keep, dtype=torch.long)

def apply_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    cls: torch.Tensor,
    nms_type: str = "standard",
    iou_thr: float = 0.5,
    diou_thr: float = 0.6,
    soft_sigma: float = 0.5,
    soft_score_thr: float = 0.001,
    soft_method: str = "gaussian",
    class_agnostic: bool = True,
    max_det: int = 300,
    min_score: float = 0.0,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    nms_type = nms_type.lower()
    assert nms_type in ("standard", "diou", "soft")

    if min_score > 0:
        m = scores >= min_score
        boxes, scores, cls, base_idx = boxes[m], scores[m], cls[m], torch.nonzero(m, as_tuple=False).squeeze(1)
    else:
        base_idx = torch.arange(scores.numel(), device=scores.device)

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    if max_det is not None and scores.numel() > max_det * 5:
        topk = min(scores.numel(), max_det * 5)
        order = scores.argsort(descending=True)[:topk]
        boxes, scores, cls, base_idx = boxes[order], scores[order], cls[order], base_idx[order]

    if class_agnostic:
        keep_local = _run_single_class_nms(boxes, scores, nms_type, iou_thr, diou_thr, soft_sigma, soft_score_thr, soft_method)
        keep = base_idx[keep_local]
        if max_det is not None and keep.numel() > max_det:
            keep = keep[scores[keep_local].argsort(descending=True)[:max_det]]
        return keep

    keep_all = []
    cls_ids = cls.to(torch.int64)
    for c in cls_ids.unique():
        mask = cls_ids == c
        b_c = boxes[mask]
        s_c = scores[mask]
        idx_c = base_idx[mask]

        keep_c_local = _run_single_class_nms(b_c, s_c, nms_type, iou_thr, diou_thr, soft_sigma, soft_score_thr, soft_method)
        keep_all.append(idx_c[keep_c_local])

    keep = torch.cat(keep_all, dim=0) if keep_all else torch.empty((0,), dtype=torch.long)
    if keep.numel() > 0 and max_det is not None and keep.numel() > max_det:
        kept_scores = scores[(base_idx.unsqueeze(1) == keep.unsqueeze(0)).any(dim=0)]  
        keep = keep[:max_det]
    return keep

def _run_single_class_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    nms_type: str,
    iou_thr: float,
    diou_thr: float,
    soft_sigma: float,
    soft_score_thr: float,
    soft_method: str,
) -> torch.Tensor:
    if nms_type == "standard":
        return standard_nms_xyxy(boxes, scores, iou_thr=iou_thr)
    if nms_type == "diou":
        return diou_nms_xyxy(boxes, scores, diou_thr=diou_thr)
    # soft
    return soft_nms_xyxy(
        boxes, scores,
        iou_thr=iou_thr,
        sigma=soft_sigma,
        score_thr=soft_score_thr,
        method=soft_method
    )