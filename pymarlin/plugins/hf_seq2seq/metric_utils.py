from rouge_score import rouge_scorer

""" Metric Functions """


def get_metric_func(metric_name):
    METRIC_MAP = {"rouge": rouge}
    return METRIC_MAP[metric_name]


def rouge(preds, labels):
    # All Rouge scores for CNN/DailyMail
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )
    agg_scores = {}

    # sum up fmeasures
    for pred, ref in zip(preds, labels):
        scores = scorer.score(pred, ref)
        for key in scores:
            if key not in agg_scores:
                agg_scores[key] = 0
            agg_scores[key] += scores[key].fmeasure

    # and divide to average
    for key in agg_scores:
        agg_scores[key] /= len(preds)

    return agg_scores
