from __future__ import print_function, absolute_import
import numpy as np
import torch


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def build_evaluate(qf, gf, method):

    m, n = qf.size(0), gf.size(0)

    if method == 'euclidean':
        q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist.addmm_(1, -2, qf, gf.t())

        q_g_dist = q_g_dist.cpu().numpy()
    elif method == 'cosine':
        q_norm = torch.norm(qf, p=2, dim=1, keepdim=True)
        g_norm = torch.norm(gf, p=2, dim=1, keepdim=True)
        qf = qf.div(q_norm.expand_as(qf))
        gf = gf.div(g_norm.expand_as(gf))
        q_g_dist = - torch.mm(qf, gf.t())

    return q_g_dist

def evaluate_reranking(qf, q_pids, q_camids, gf, g_pids, g_camids, ranks, cal_method):

    q_g_dist = build_evaluate(qf, gf, cal_method)

    print("Computing CMC and mAP")
    be_cmc, be_mAP = evaluate(q_g_dist, q_pids, g_pids, q_camids, g_camids)

    print("feature Results ----------")
    print("mAP: {:.1%}".format(be_mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, be_cmc[r - 1]))
    print("------------------")
    print()

    return be_cmc, q_g_dist

if __name__ == "__main__":
    cam1 = torch.Tensor\
            ([[0.0401, 0.0401, 0.1961, 0.1461, 0.1560, 0.1422, 0.1180, 0.3322],
            [0.0401, 0.0401, 0.1182, 0.5560, 0.5209, 0.1101, 0.1988, 0.2415],
            [0.0521, 0.1526, 0.6284, 0.6227, 0.6323, 0.1592, 0.1260, 0.2741],
            [0.0345, 0.8586, 0.6726, 0.9587, 0.8517, 0.9514, 0.1826, 0.2457],
            [0.2329, 0.8595, 0.6809, 0.4199, 0.6289, 0.7401, 0.1904, 0.1973],
            [0.3420, 0.6701, 0.5793, 0.4116, 0.6841, 0.6966, 0.1795, 0.1178],
            [0.2506, 0.5005, 0.5541, 0.5789, 0.7487, 0.6012, 0.4126, 0.1420],
            [0.2447, 0.4981, 0.5177, 0.6408, 0.6534, 0.5642, 0.4797, 0.1763],
            [0.2185, 0.4611, 0.6022, 0.6298, 0.4598, 0.4793, 0.1735, 0.1092],
            [0.1174, 0.2513, 0.5548, 0.4883, 0.4631, 0.4643, 0.1294, 0.1686],
            [0.1491, 0.2936, 0.5029, 0.4680, 0.4591, 0.1939, 0.1658, 0.1522],
            [0.1224, 0.1331, 0.5650, 0.4009, 0.4313, 0.1271, 0.0633, 0.1179],
            [0.1046, 0.2434, 0.4011, 0.3123, 0.3660, 0.1951, 0.1026, 0.1236],
            [0.1110, 0.2048, 0.0728, 0.0583, 0.1590, 0.1766, 0.1642, 0.1427],
            [0.1791, 0.2176, 0.0584, 0.1921, 0.1322, 0.0937, 0.1775, 0.1400],
            [0.1521, 0.1826, 0.1989, 0.1748, 0.1576, 0.1498, 0.1295, 0.1905]])
    a = np.random.rand(3, 2)
    b = np.random.rand(4, 2)
    q_g_dist = np.power(a, 2).sum(1, keepdims=True).repeat(4, axis=1) + \
               np.power(b, 2).sum(1, keepdims=True).repeat(3, axis=1).t()
    q_g_dist = q_g_dist - 2 * a.matmul(b.t())

    a = torch.Tensor(a)
    b = torch.Tensor(b)
    q_g_dist2 = torch.pow(a, 2).sum(dim=1, keepdim=True).expand(3, 4) + \
               torch.pow(b, 2).sum(dim=1, keepdim=True).expand(4, 3).t()
    q_g_dist2.addmm_(1, -2, a, b.t())
