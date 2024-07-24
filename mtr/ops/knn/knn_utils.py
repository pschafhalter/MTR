# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Li Jiang, Shaoshuai Shi 
# All Rights Reserved


import torch
import torch.nn as nn
from torch.autograd import Function

from . import knn_cuda


class KNNBatch(Function):
    @staticmethod
    def forward(ctx, xyz, query_xyz, batch_idxs, query_batch_offsets, k):
        '''
        :param ctx:
        :param xyz: (n, 3) float
        :param query_xyz: (m, 3), float
        :param batch_idxs: (n) int
        :param query_batch_offsets: (B+1) int, offsets[-1] = m
        :param k: int
        :return: idx (n, k)
        '''

        n = xyz.size(0)
        m = query_xyz.size(0)
        assert k <= m
        assert xyz.is_contiguous() and xyz.is_cuda
        assert query_xyz.is_contiguous() and query_xyz.is_cuda
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda
        assert query_batch_offsets.is_contiguous() and query_batch_offsets.is_cuda

        idx = torch.cuda.IntTensor(n, k).zero_()

        knn_cuda.knn_batch(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, k)

        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None
    

knn_batch = KNNBatch.apply


class KNNBatchMlogK(Function):
    @staticmethod
    def forward(ctx, xyz, query_xyz, batch_idxs, query_batch_offsets, k):
        '''
        :param ctx:
        :param xyz: (n, 3) float
        :param query_xyz: (m, 3), float
        :param batch_idxs: (n) int
        :param query_batch_offsets: (B+1) int, offsets[-1] = m
        :param k: int
        :return: idx (n, k)
        '''
        n = xyz.size(0)
        m = query_xyz.size(0)
        # assert k <= m
        assert xyz.is_contiguous() and xyz.is_cuda
        assert query_xyz.is_contiguous() and query_xyz.is_cuda
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda
        assert query_batch_offsets.is_contiguous() and query_batch_offsets.is_cuda
        assert k <= 128
        idx = torch.cuda.IntTensor(n, k).zero_()

        knn_cuda.knn_batch_mlogk(xyz, query_xyz, batch_idxs, query_batch_offsets, idx, n, m, k)

        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None
   
knn_batch_mlogk = KNNBatchMlogK.apply

class KNNBatchPython(Function):
    """Python kNN implementation which does not rely on CUDA kernels.

    Differences in the results from the CUDA implementation are due to different
    tie-breaking strategies for equidistant points.
    """
    @staticmethod
    def forward(ctx, xyz, query_xyz, batch_idxs, query_batch_offsets, k):
        result = torch.zeros((xyz.shape[0], k), dtype=torch.int32, device=xyz.device)

        for start, end in zip(query_batch_offsets, query_batch_offsets[1:]):
            xyz_view = xyz[start:end]
            query_xyz_view = query_xyz[start:end]
            distances = torch.cdist(xyz_view, query_xyz_view)
            values, indices = torch.sort(distances)
            result[start:end] = indices[:, :k]

        return result

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None

knn_batch_python = KNNBatchPython.apply


def knn_batch_python_debug(xyz, query_xyz, batch_idxs, query_batch_offsets, k, ref=None):
    """KNN in python that compares to a reference implementation.

    Any differences to the CUDA reference implementation is due to differences in
    breaking ties between equidistant points.
    """

    result = torch.zeros((xyz.shape[0], k), dtype=torch.int32, device=xyz.device)

    for start, end in zip(query_batch_offsets, query_batch_offsets[1:]):
        xyz_view = xyz[start:end]
        query_xyz_view = query_xyz[start:end]
        distances = torch.cdist(xyz_view, query_xyz_view)
        values, indices = torch.sort(distances)
        if ref is not None:
            for i in range(indices.shape[0]):
                r_pred = set(indices[i, :k].cpu().tolist())
                r_true = set(ref[start:end][i].cpu().tolist())
                if r_pred != r_true:
                    print(r_pred - r_true, r_true - r_pred)
                    diff_vals = (r_pred - r_true) | (r_true - r_pred)
                    diff_dict = {}
                    for x, y in zip(indices[i].cpu().tolist(), values[i].cpu().tolist()):
                        if x in diff_vals:
                            diff_dict[x] = y
                    print(diff_dict)
        result[start:end] = indices[:, :k]

    return result


def compare_knn_results(result1, result2) -> bool:
    if result1.shape != result2.shape:
        return False
    result = True
    for i in range(result1.shape[0]):
        r1 = set(result1[i].cpu().tolist())
        r2 = set(result2[i].cpu().tolist())
        if r1 != r2:
            print(r1 - r2, r2 - r1)
            result = False
    return result