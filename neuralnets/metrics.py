import numpy as np


def _check_transfer(predictions, labels):
    assert predictions.ndim == 1
    assert labels.ndim == 1
    return predictions.astype(np.int32), labels.astype(np.int32)


def accuracy(predictions, labels):
    assert predictions.shape == labels.shape
    p, l = predictions.astype(np.int32), labels.astype(np.int32)
    return np.where(p == l, 1., 0.).mean()


def roc(logits, labels, num_thresholds):
    # Receiver Operating Characteristic curve
    assert logits.ndim == 1
    assert labels.ndim == 1
    if labels.dtype != np.int32:
        labels = labels.astype(np.int32)
    zeros, ones = np.zeros_like(logits, dtype=np.int32), np.ones_like(logits, dtype=np.int32)
    roc_data = np.empty((num_thresholds, 2), dtype=np.float32)
    for i, threshold in enumerate(np.linspace(0, 1, num=num_thresholds, endpoint=True, dtype=np.float32)):
        if threshold == 0:
            roc_data[i, :] = [0, 0]
            continue
        if threshold == 1:
            roc_data[i, :] = [1, 1]
            break
        p = np.where(logits < threshold, zeros, ones)
        tpr = true_pos_rate(p, labels)
        fpr = false_pos_rate(p, labels)
        roc_data[i, :] = [fpr, tpr]
    roc_data = np.sort(roc_data, axis=0)
    return roc_data


def auc(logits, labels, num_thresholds=200):
    # Area Under the Curve: for binary classifier who outputs only [0,1]
    roc_data = roc(logits, labels, num_thresholds)
    diff = np.diff(roc_data, axis=0)
    _auc = (diff[:, 0] * diff[:, 1] / 2 + roc_data[:-1, 1] * diff[:, 0]).sum()  # areas of triangles + rectangle
    return _auc


def true_pos(predictions, labels):
    return np.count_nonzero((predictions == 1) & (labels == 1))


def true_neg(predictions, labels):
    return np.count_nonzero((predictions == 0) & (labels == 0))


def false_pos(predictions, labels):
    return np.count_nonzero((predictions == 1) & (labels == 0))


def false_neg(predictions, labels):
    return np.count_nonzero((predictions == 0) & (labels == 1))


def true_pos_rate(predictions, labels):
    # TP / (FP + FN) = TP / P
    p, l = predictions, labels
    return true_pos(p, l) / np.count_nonzero(labels)


def false_pos_rate(predictions, labels):
    # FP / (FP + TN) = FP / N
    p, l = predictions, labels
    return false_neg(p, l) / (len(labels) - np.count_nonzero(labels))

