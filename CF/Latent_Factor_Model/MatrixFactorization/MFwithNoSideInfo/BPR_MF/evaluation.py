import numpy as np
import torch

def precision_at_k(r, k):
    r = np.asarray(r)[:k]
    if k == 0:
        return 0.0
    return np.sum(r) / k

def recall_at_k(r, k, true_positives):
    r = np.asarray(r)[:k]
    true_positives = np.asarray(true_positives)[:k]
    n_true_positives = np.sum(true_positives)
    return np.sum(true_positives) / n_true_positives if n_true_positives != 0 else 0.0


def f1_at_k(r, k, all_positives):
    p_at_k = precision_at_k(r, k)
    r_at_k = recall_at_k(r, k, all_positives)
    if p_at_k + r_at_k == 0:
        return 0.0
    return 2 * p_at_k * r_at_k / (p_at_k + r_at_k)

def ndcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size == 0:
        return 0.0
    dcg = np.sum((2 ** r - 1) / np.log2(np.arange(2, r.size + 2)))
    idcg = np.sum((2 ** np.sort(r)[::-1] - 1) / np.log2(np.arange(2, r.size + 2)))
    return dcg / idcg if idcg > 0 else 0.0

def compute_metrics(model, test_data, k=5):
    ndcg_scores = []
   
    f1_scores = []
    for u in test_data:
        u_likes = test_data[u][0]
        u_dislikes = test_data[u][1]
        candidates = u_likes + u_dislikes
        labels = [1] * len(u_likes) + [0] * len(u_dislikes)
        scores = model.predict(u, candidates)
        ranked_items = [(score, label) for score, label in zip(scores, labels)]
    
        ranked_items.sort(key=lambda x: x[0], reverse=True)

        true_positives = [1 if label == 1 and score >= 0.5 else 0 for score, label in ranked_items]

        top_items = [x for x in ranked_items if x[0] >= 0.5]
        ndcg_score = ndcg_at_k([x[1] for x in ranked_items], k)
        
        
        
        f1_score = f1_at_k([x[1] for x in top_items], min(10, len(top_items)), true_positives)
        
        ndcg_scores.append(ndcg_score)
        f1_scores.append(f1_score)
    return np.mean(ndcg_scores), np.mean(f1_scores)


def evaluate(model, test_loader, test_data, k=5):
    model.eval()
    hits = 0
    total = 0
    with torch.no_grad():
        for users, items_pos, items_neg in test_loader:
            users, items_pos, items_neg = users.long(), items_pos.long(), items_neg.long()
            y_ui = torch.sum(model.U[users] * model.V[items_pos], dim=1)
            y_uj = torch.sum(model.U[users] * model.V[items_neg], dim=1)
            hits += torch.sum(y_ui > y_uj).item()
            total += len(users)
    
    accuracy = hits / total
    ndcg_k, f1_k = compute_metrics(model, test_data, k=k)
    print(f'Accuracy: {accuracy:.4f} - NDCG@{k}: {ndcg_k:.4f} - F1@{10}: {f1_k:.4f}')


