"""Observed feedback metrics and monotone, validation-only probability calibration."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, rel_entr
from sklearn.metrics import roc_auc_score


def _binary_inputs(labels, logits):
    labels = np.asarray(labels, dtype=np.float64); logits = np.asarray(logits, dtype=np.float64)
    if (labels.ndim != 1 or not len(labels) or logits.shape != labels.shape
            or not np.isin(labels, [0,1]).all() or not np.isfinite(logits).all()):
        raise ValueError('Aligned nonempty binary labels and finite logits required')
    return labels, logits


def binary_terms(labels, logits):
    labels, logits = _binary_inputs(labels, logits)
    p = expit(logits)
    return np.column_stack([labels, np.logaddexp(0,logits)-labels*logits, (p-labels)**2])


def binary_metrics(labels, logits, weights=None, bins=15):
    labels, logits = _binary_inputs(labels, logits)
    if not isinstance(bins, (int,np.integer)) or bins < 1:
        raise ValueError('Positive integer calibration bin count required')
    weighted = weights is not None
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights,dtype=np.float64)
    if weights.shape != labels.shape or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError('Finite nonnegative aligned weights with positive sum required')
    total = float(weights.sum()); norm = weights/total; p = expit(logits)
    terms = binary_terms(labels,logits)
    positive_mass = float(norm@labels)
    # A normalized weighted mean can round below 1 for a one-class group.
    # Check actual class mass, not floating-point equality of the mean.
    has_both = weights[labels==1].sum()>0 and weights[labels==0].sum()>0
    auc = float(roc_auc_score(labels,logits,sample_weight=weights)) if has_both else None
    which = np.minimum((p*bins).astype(int),bins-1)
    calibration = []; ece=0.
    for k in range(bins):
        mask=which==k; mass=float(weights[mask].sum()); rows=int(mask.sum())
        pred=float(weights[mask]@p[mask]/mass) if mass else None
        observed=float(weights[mask]@labels[mask]/mass) if mass else None
        if mass: ece+=mass/total*abs(pred-observed)
        calibration.append({'bin':k,'left':k/bins,'right':(k+1)/bins,'rows':rows,
                            'weight_mass':mass,'mean_probability':pred,'positive_rate':observed})
    return {'rows':len(labels),'positive_weight_rows':int((weights>0).sum()),
            'positives':int(labels.sum()),'positive_rate':positive_mass,
            'auc':auc,'logloss':float(norm@terms[:,1]),'brier':float(norm@terms[:,2]),
            'ece':float(ece),'ece_bins':bins,'weighted':weighted,'total_weight':total,
            'calibration':calibration}


def fit_platt(logits, labels):
    labels, logits = _binary_inputs(labels,logits)
    if len(np.unique(labels)) != 2:
        raise ValueError('Calibration needs both observed outcomes')
    def objective(theta):
        z=theta[0]*logits+theta[1]
        residual=expit(z)-labels
        loss=np.mean(np.logaddexp(0,z)-labels*z)
        return float(loss),np.array([np.mean(residual*logits),residual.mean()])
    fitted=minimize(objective,np.array([1.,0.]),jac=True,method='L-BFGS-B',
                    bounds=[(1e-6,20.),(-30.,30.)],options={'ftol':1e-12,'gtol':1e-9,'maxiter':200})
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError('Validation-only Platt fit did not converge: '+str(fitted.message))
    return {'scale':float(fitted.x[0]),'intercept':float(fitted.x[1]),
            'fit_rows':len(labels),'fit_positives':int(labels.sum()),
            'fit_logloss_before':objective([1.,0.])[0],'fit_logloss_after':float(fitted.fun),
            'monotone':True,'optimizer':'L-BFGS-B; scale in [1e-6,20], intercept in [-30,30]',
            'iterations':int(fitted.nit)}


def apply_platt(logits, model):
    logits=np.asarray(logits,dtype=np.float64)
    if not np.isfinite(logits).all() or not np.isfinite(model['scale']) or model['scale']<=0 or not np.isfinite(model['intercept']):
        raise ValueError('Finite logits and positive monotone calibration scale required')
    adjusted=model['scale']*logits+model['intercept']
    if not np.isfinite(adjusted).all():raise ValueError('Calibrated logits overflow')
    return adjusted


def shared_stratum_weights(first_users, first_tabs, second_users, second_tabs):
    u1=np.asarray(first_users);t1=np.asarray(first_tabs);u2=np.asarray(second_users);t2=np.asarray(second_tabs)
    if (u1.ndim!=1 or u2.ndim!=1 or t1.shape!=u1.shape or t2.shape!=u2.shape
            or not len(u1) or not len(u2) or u1.dtype.kind not in 'iu' or u2.dtype.kind not in 'iu'):
        raise ValueError('Aligned nonempty integer users/tabs required')
    pairs=np.column_stack([np.concatenate([u1,u2]),np.concatenate([t1,t2])])
    _,inv=np.unique(pairs,axis=0,return_inverse=True);i1=inv[:len(u1)];i2=inv[len(u1):]
    n=int(inv.max()+1);c1=np.bincount(i1,minlength=n);c2=np.bincount(i2,minlength=n)
    shared=(c1>0)&(c2>0);count=int(shared.sum())
    if not count:raise ValueError('No shared user-tab strata')
    w1=shared[i1]/(c1[i1]*count);w2=shared[i2]/(c2[i2]*count)
    return w1,w2,{'shared_strata':count,'first_included_rows':int(shared[i1].sum()),
                  'second_included_rows':int(shared[i2].sum()),
                  'target':'equal weight for each shared (raw user_id, tab) stratum',
                  'limits':'Items and request times are not matched; this is not causal adjustment'}


def js_divergence_counts(first, second):
    first=np.asarray(first,dtype=np.float64);second=np.asarray(second,dtype=np.float64)
    if first.shape!=second.shape or first.ndim!=1 or not np.isfinite(first).all() or not np.isfinite(second).all() or (first<0).any() or (second<0).any() or first.sum()<=0 or second.sum()<=0:
        raise ValueError('Aligned nonnegative finite counts with positive totals required')
    p=first/first.sum();q=second/second.sum();m=(p+q)/2
    return float((rel_entr(p,m).sum()+rel_entr(q,m).sum())/(2*np.log(2)))
