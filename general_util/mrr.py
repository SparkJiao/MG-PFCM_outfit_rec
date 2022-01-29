"""
Calculate the MRR metric.
"""


def get_rank(pos_score, neg_scores):
    rank = 1
    for score in neg_scores:
        if score > pos_score:
            rank += 1
    return rank


def get_mrr(scores_list):
    mrr = 0
    for scores in scores_list:
        pos_score = scores[0]
        neg_scores = scores[1:]
        rank = get_rank(pos_score, neg_scores)
        mrr += 1.0 / rank
    return mrr / len(scores_list)
