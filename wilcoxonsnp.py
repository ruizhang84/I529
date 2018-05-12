import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import wilcoxon
# from sklearn.utils import resample
from readsnp import readdata

def rf_score(X, y, k=10, r=30):
    """
        input: X, y datastes, the number of trees in random forest
        output: the important score of each features by random forest
                score (n_features, r)
        ---
        implements the bootstrapping procedure;
        builds decision trees (i.e., random forest);
        computes the important scores from random forest.
    """
    _, n_feature = X.shape
    score = np.zeros((r, n_feature))
    for i in range(r):
        # X_train, y_train = resample(X, y)  

        clf = RandomForestClassifier(n_estimators=k, random_state=i)
        clf.fit(X, y)
        score[i, ] = clf.feature_importances_
    return np.swapaxes(score,0,1)

def process_scores(scores, n_snp):
    """
        input: important scores
        output: a list k-length scores for each SNP
                and one k-length scores for maximal shadow SNP
        --
        split the scores into SNP and shadow SNP
    """
    # _, k = scores.shape

    score_test = scores[:n_snp]
    score_shadow = np.mean(scores[n_snp:], axis=0)
    # score_shadow = np.zeros(k)
    # for i in range(n_snp):
    #     score_shadow = np.max([score_shadow, scores[n_snp+i]], axis=0)
    return score_test, score_shadow

def gen_snp(X, y, n_snp, k=200, r=30, p=0.05):
    """
        input:  training data X and y
        output: a list of SNP
        --
        generates a list of SNP
        1) performs a wilcoxon test,
        2) filters snp by a statistic threshold p
    """
    scores = rf_score(X, y, k, r)
    score_test, score_shadow = process_scores(scores, n_snp)
    snp = []
    snp_info = []
    for i in range(len(score_test)):
        st = score_test[i]
        if np.sum(st) == 0:
            continue
        if np.mean(st) < np.mean(score_shadow):
            continue
        _, val = wilcoxon(st, score_shadow)
        if val/2.0 < p:
            snp.append(i)
            snp_info.append((-np.mean(st), i, val/2.0))
    return snp, snp_info
