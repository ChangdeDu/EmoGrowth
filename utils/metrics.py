import torch
import numpy as np
def evaluation(scores_, targets_):
    n, n_class = scores_.shape
    Na, Nc, Np, Ng = torch.zeros(n_class), torch.zeros(n_class), torch.zeros(n_class), torch.zeros(n_class)
    for k in range(n_class):
        scores = scores_[:, k]
        targets = targets_[:, k]
        Ng[k] = torch.sum(targets == 1) #实际为正
        Np[k] = torch.sum(scores >= 0)  #预测为正
        Nc[k] = torch.sum(targets * (scores >= 0)) #正确预测为正
        Na[k] = (torch.sum((scores < 0) * (targets == 0)) + Nc[k]) / n
    Np[Np == 0] = 1 #防止除0
    OP = torch.sum(Nc) / torch.sum(Np)
    OR = torch.sum(Nc) / torch.sum(Ng)
    OF1 = (2 * OP * OR) / (OP + OR)
    MA = torch.sum(Na) / n_class
    CP = torch.sum(Nc / Np) / n_class
    CR = torch.sum(Nc / Ng) / n_class
    CF1 = (2 * CP * CR) / (CP + CR)


    return CP,CR,CF1,OP,OR,OF1

def evaluation_topk(scores_, targets_,k):
    n, n_class = scores_.shape
    scores_new = torch.zeros((n, n_class)) - 1
    index = scores_.topk(k, 1, True, True)[1].cpu().numpy()
    for i in range(n):
        for ind in index[i]:
            scores_new[i, ind] = 1 if scores_[i, ind] >= 0 else -1
    scores_ = scores_new.cuda()
    Na, Nc, Np, Ng = torch.zeros(n_class), torch.zeros(n_class), torch.zeros(n_class), torch.zeros(n_class)
    for k in range(n_class):
        scores = scores_[:, k]
        targets = targets_[:, k]
        Ng[k] = torch.sum(targets == 1) #实际为正
        Np[k] = torch.sum(scores >= 0)  #预测为正
        Nc[k] = torch.sum(targets * (scores >= 0)) #正确预测为正
        Na[k] = (torch.sum((scores < 0) * (targets == 0)) + Nc[k]) / n
    Np[Np == 0] = 1 #防止除0
    OP = torch.sum(Nc) / torch.sum(Np)
    OR = torch.sum(Nc) / torch.sum(Ng)
    OF1 = (2 * OP * OR) / (OP + OR)
    MA = torch.sum(Na) / n_class
    CP = torch.sum(Nc / Np) / n_class
    CR = torch.sum(Nc / Ng) / n_class
    CF1 = (2 * CP * CR) / (CP + CR)


    return CP,CR,CF1,OP,OR,OF1

def average_precision(scores_, targets_):
    n, n_class = scores_.shape
    ap = torch.zeros(n_class)
    for k in range(n_class):
        scores = scores_[:, k]
        targets = targets_[:, k]
        sorted, indices = torch.sort(scores, dim=0, descending=True)
        pos_count = 0. #真实为正
        total_count = 0. #所有预测为正
        precision_at_i = 0.
        for i in indices:
            label = targets[i]
            total_count += 1
            if label == 0:
                continue
            else:
                pos_count += 1
                precision_at_i += pos_count / total_count
        ap[k] = precision_at_i / pos_count
    return ap,torch.mean(ap)



FMT = '%.4f'
def AveragePrecision(outputs, true_labels):
    m, q = true_labels.shape
    ap = 0
    all_zero_m = 0
    for i in range(m):
        rel_lbl = np.count_nonzero(true_labels[i])
        if rel_lbl != 0:
            rel_lbl_idx = np.where(true_labels[i] == 1)[0]
            tmp_out = outputs[i]
            sort_idx = np.argsort(-tmp_out)
            cnt = 0
            for j in rel_lbl_idx:
                t = np.argwhere((tmp_out >= outputs[i, j]) == True)
                cntt = len(np.intersect1d(t, rel_lbl_idx))
                pre_lbl_rank = np.where(sort_idx[:] == j)[0][0] + 1
                cnt += cntt / pre_lbl_rank
            cnt /= rel_lbl
        else:
            all_zero_m += 1
            cnt = 0
        ap += cnt
    ap /= (m - all_zero_m)
    return float(FMT % ap)

def RankingLoss(outputs, true_labels):
    m, q = true_labels.shape
    rl = 0
    all_zero_m = 0
    for i in range(m):
        rel_lbl = np.count_nonzero(true_labels[i])
        if rel_lbl != 0:
            tmp_out = outputs[i, :]
            sort_idx = np.argsort(-tmp_out)
            tmp_true = true_labels[i, :][sort_idx]
            n_zero = 0
            rl_ins = 0
            for j in range(q):
                if tmp_true[j] == 0:
                    n_zero += 1
                elif tmp_true[j] == 1:
                    rl_ins += n_zero
            rl += rl_ins / (rel_lbl * (q - rel_lbl)+1e-5)
        else:
            all_zero_m += 1
    rl = rl / (m - all_zero_m)
    return float(FMT % rl)

def Coverage(outputs, true_labels):
    m, q = true_labels.shape
    cov = 0
    for i in range(m):
        tmp_out = outputs[i, :]
        sort_idx = np.argsort(-tmp_out)
        tmp_true = true_labels[i, :][sort_idx]
        if 0 != np.sum(tmp_true):
            cov += np.max(np.where(tmp_true == 1))
    return float(FMT % (cov / m / q))

def OneError(outputs, true_labels):
    m, q = true_labels.shape
    index = np.argmax(outputs, axis=1)
    true_labels = true_labels.reshape(1, m * q)
    index = [i * q for i in range(m)] + index
    oe = np.sum(true_labels[:, index] != 1) / m
    return float(FMT % oe)

def HammingLoss(outputs, true_labels):
    pre_labels = np.array(outputs > 0, dtype=np.int)
    m, q = true_labels.shape
    miss_label = np.sum((pre_labels == true_labels) == False)
    hl = miss_label / (m * q)
    return float(FMT % hl)

def MacroF1(outputs, true_labels):
    pre_labels = np.array(outputs > 0, dtype=np.int)
    true_labels = true_labels.astype(int)
    m, q = true_labels.shape
    maf = 0
    for i in range(q):
        tp = np.sum(((pre_labels[:, i]) & (true_labels[:, i])) == True)
        fp = np.sum((pre_labels[:, i] & (1 - true_labels[:, i])) == True)
        fn = np.sum(((1 - pre_labels[:, i]) & true_labels[:, i]) == True)
        if tp + fp + fn == 0:
            tmp_maf = 0
        else:
            tmp_maf = (2 * tp) / (2 * tp + fp + fn)
        maf += tmp_maf
    return float(FMT % (maf / q))


def MicroF1(outputs, true_labels):
    pre_labels = np.array(outputs > 0, dtype=np.int)
    true_labels = true_labels.astype(int)
    tp = np.sum(((pre_labels) & (true_labels)) == True)
    fp = np.sum((pre_labels & (1 - true_labels)) == True)
    fn = np.sum(((1 - pre_labels) & true_labels) == True)
    if tp + fp + fn == 0:
        mif = 0
    else:
        mif = (2 * tp) / (2 * tp + fp + fn)
    return float(FMT % mif)

def all_metrics(outputs, true_labels):
    metrics_name = ['hamming_loss', 'avg_precision', 'one_error', 'ranking_loss', 'coverage', 'macrof1', 'microf1']
    hamming_loss = HammingLoss(outputs, true_labels)
    avg_precision = AveragePrecision(outputs, true_labels)
    one_error = OneError(outputs, true_labels)
    ranking_loss = RankingLoss(outputs, true_labels)
    coverage = Coverage(outputs, true_labels)
    macrof1 = MacroF1(outputs, true_labels)
    microf1 = MicroF1(outputs, true_labels)
    metrics_res = [hamming_loss, avg_precision, one_error, ranking_loss, coverage, macrof1, microf1]
    return list(zip(metrics_name, metrics_res))