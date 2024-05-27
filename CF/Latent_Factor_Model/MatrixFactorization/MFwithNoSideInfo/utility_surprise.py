from datetime import datetime
import numpy as np
import random
from collections import defaultdict
from math import log2

my_seed = 15
random.seed(my_seed)
np.random.seed(my_seed)

def get_ratings(predictions):
    actual = np.array([pred.r_ui for pred in predictions])
    pred = np.array([pred.est for pred in predictions])
    return actual, pred
def get_errors(predictions):
    actual, pred = get_ratings(predictions)
    rmse = np.sqrt(np.mean((pred - actual)**2))
    mape = np.mean(np.abs(actual-pred)/actual) * 100
    return rmse, mape
def evaluation(predictions, k=5, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions = defaultdict()
    recalls = defaultdict()
    ndcgs = defaultdict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        dcg = sum(true_r / log2(i+2) if true_r >= threshold else 0 for i,(_, true_r) in enumerate(user_ratings[:k]))
        sorted_true = sorted(user_ratings, key = lambda x: x[1], reverse = True)
        idcg = sum(true_r / log2(i+2) if true_r >= threshold else 0 for i,(_, true_r) in enumerate(sorted_true[:k]))
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:10])
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:10]
        )
        ndcgs[uid] = dcg/idcg if idcg!= 0 else 0
        precisions[uid]= n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid]= n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    precision = sum(prec for prec in precisions.values()) / len(precisions)
    recall = sum(rec for rec in recalls.values()) / len(recalls)
    ndcg = sum(nd for nd in ndcgs.values()) / len(ndcgs)
    f1_score = sum((2*precisions[i]*recalls[i]/(precisions[i]+ recalls[i])) if precisions[i] +recalls[i] != 0 else 0 for i in (precisions.keys()))/len(precisions)

    return precision, recall, f1_score , ndcg
def run_surprise(algo, trainset, testset, verbose=True): 
    start = datetime.now()
    train = dict()
    test = dict()
    st = datetime.now()
    print('Training the model...')
    algo.fit(trainset)
    print('Done. time taken : {} \n'.format(datetime.now()-st))
    st = datetime.now()
    print('Evaluating the model with train data..')
    train_preds = algo.test(trainset.build_testset())
    precision, recall,f1, ndcg = evaluation(train_preds)
    train_rmse, train_mape = get_errors(train_preds)
    print('time taken : {}'.format(datetime.now()-st))
    if verbose:
        print('-'*15)
        print('Train Data')
        print('-'*15)
        print("RMSE : {}\n\nMAPE : {}\n".format(train_rmse, train_mape))
        print()
    if verbose:
        print('adding train results in the dictionary..')
    train['rmse'] = train_rmse
    train['mape'] = train_mape
    train['recall'] = recall
    train['precision'] = precision
    train['f1'] = f1
    st = datetime.now()
    print('\nEvaluating for test data...')
    test_preds = algo.test(testset)
    test_rmse, test_mape = get_errors(test_preds)
    precision, recall,f1,ndcg = evaluation(test_preds)
   
    print('time taken : {}'.format(datetime.now()-st))
    if verbose:
        print('-'*15)
        print('Test Data')
        print('-'*15)
        print("RMSE : {}\n\nMAPE : {}\n".format(test_rmse, test_mape))
        print("Precision@10 : {}\n\nRecall@10 : {}\n\nF1@10 : {}\n\nNDCG@5: {}".format(precision, recall, f1, ndcg))
    if verbose:
        print('storing the test results in test dictionary...')
    test['rmse'] = test_rmse
    test['mae'] = test_mape
    test['recall'] = recall
    test['precision'] = precision
    test['f1'] = f1
    test['ndcg'] = ndcg

    print('\n'+'-'*45)
    print('Total time taken to run this algorithm :', datetime.now() - start)
    return train,test