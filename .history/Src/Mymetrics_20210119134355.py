from sklearn.metrics import cohen_kappa_score
import sys
sys.path.append("./")
from common import *

def quadratic_weighted_kappa(y_hat, y):
    return cohen_kappa_score(y_hat, y, weights="quadratic")

#-------------------Metric-----------------------#
def np_dice_score(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)
    ths = 0.5
    union = torch.zeros(1)
    overlap = torch.zeros(1)
    # for i,th in enumerate(ths):
    pt = (p>ths)
    tt = (t>ths)
    union = pt.sum() + tt.sum()
    overlap = (pt*tt).sum()

    dice = 2*overlap/(union+0.001)
    return dice, ths

def np_accuracy(probability, mask):
    p = probability.reshape(-1)
    t = mask.reshape(-1)
    p = p>0.5
    t = t>0.5
    tp = (p*t).sum()/((t).sum()+1e6)
    tn = ((1-p)*(1-t)).sum()/((1-t).sum()+1e6)
    return tp, tn