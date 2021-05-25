import os
import numpy as np
import math
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    accuracy_score,
)

""" Metric Functions """


def get_metric_func(metric_name):
    METRIC_MAP = {
        "acc": simple_accuracy,
        "acc_and_f1": acc_and_f1,
        "pcc_and_scc": pearson_and_spearman,
        "mcc": mcc,
    }
    return METRIC_MAP[metric_name]


def mcc(labels, preds):
    return {"mcc": matthews_corrcoef(labels, preds)}


def simple_accuracy(labels, preds):
    return {"acc": accuracy_score(preds, labels)}


def acc_and_f1(labels, preds, average="weighted", target_labels=None):
    f1 = f1_score(y_true=labels, y_pred=preds, average=average, labels=target_labels)
    precision = precision_score(
        y_true=labels, y_pred=preds, average=average, labels=target_labels
    )
    recall = recall_score(
        y_true=labels, y_pred=preds, average=average, labels=target_labels
    )
    metrics_dict = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
    metrics_dict.update(simple_accuracy(labels, preds))
    return metrics_dict


def pearson_and_spearman(labels, preds):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }
