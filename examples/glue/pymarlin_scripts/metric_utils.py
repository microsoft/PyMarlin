import os
import numpy as np
import math
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, classification_report, accuracy_score

from pymarlin.utils.logger.logging_utils import getlogger
logger = getlogger(__name__)

''' Metric Functions 
    Add a new metric function here and import it for your task.'''

def mcc(labels, preds):
    return {"mcc": matthews_corrcoef(labels, preds)}

def simple_accuracy(labels, preds):
    return {"acc": accuracy_score(preds, labels)}

def acc_and_f1(labels, preds, average='binary', target_labels=None):
    f1 = f1_score(y_true=labels, y_pred=preds, average=average, labels=target_labels)
    precision = precision_score(y_true=labels, y_pred=preds, average=average, labels=target_labels)
    recall = recall_score(y_true=labels, y_pred=preds, average=average, labels=target_labels)
    metrics_dict = {"f1": f1, "precision": precision, "recall": recall,}
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

def getDiscountedCumulativeGain(sortedRelevance, k):
    dcg = 0.0
    p = len(sortedRelevance)
    if k > 0:
        p = min(p, k)
    i = 1
    for r in sortedRelevance[:p]:
        dcg += ((math.pow(2,r) - 1) / math.log2(i + 1))
        i += 1
    return dcg

def ndcgK(k, orig_list,sorted_list):
    '''
    orig_list is list of relvance scores as output by model
    sorted_list is sorted relevnace by label in reverse=True
    '''  
    ndcgK = 0.0
    dcgK = getDiscountedCumulativeGain(orig_list, k)
    idcgK = getDiscountedCumulativeGain(sorted_list, k)
    if idcgK !=0 :
        ndcgK = dcgK/idcgK
    return {"ndcgK":ndcgK}
