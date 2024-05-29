from math import log2
from collections import defaultdict
import numpy as np
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def evaluation(predictions, true_labels,user_idx, k=5, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, est, true_r in zip(user_idx, predictions, true_labels):
        user_est_true[uid].append((est, true_r))

    precisions = defaultdict()
    recalls = defaultdict()
    ndcgs = defaultdict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        dcg = sum(true_r / log2(i + 2) if true_r >= threshold else 0 for i, (_, true_r) in enumerate(user_ratings[:k]))
        sorted_true = sorted(user_ratings, key=lambda x: x[1], reverse=True)
        idcg = sum(true_r / log2(i + 2) if true_r >= threshold else 0 for i, (_, true_r) in enumerate(sorted_true[:k]))

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:10])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:10])

        ndcgs[uid] = dcg / idcg if idcg != 0 else 0
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)
    ndcg = sum(nd for nd in ndcgs.values()) / len(ndcgs)
    f1_score = sum((2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i])) if precisions[i] + recalls[i] != 0 else 0 for i in precisions.keys()) / len(precisions)

    return precision, recall, f1_score, ndcg