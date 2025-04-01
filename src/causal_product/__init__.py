#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import torch
from  multiprocessing.dummy import Pool

from .causal_product_cpu import causal_dot_product as causal_dot_product_cpu, \
                                 causal_dot_backward as causal_dot_backward_cpu


from .causal_product_numerator_cpu import causal_dot_product as causal_dot_numerator_product_cpu, \
                                 causal_dot_backward as causal_dot_numerator_backward_cpu


causal_dot_numerator_product_cuda, causal_dot_numerator_backward_cuda=None,None
causal_dot_product_cuda, causal_dot_backward_cuda = None, None

if torch.cuda.is_available():
    from .causal_product_cuda import causal_dot_product as causal_dot_product_cuda, \
                                     causal_dot_backward as causal_dot_backward_cuda

    from .causal_product_numerator_cuda import \
        causal_dot_numerator_product as causal_dot_numerator_product_cuda, \
        causal_dot_numerator_backward as causal_dot_numerator_backward_cuda


def causal_dot_product(Q, K, V, tq, tkv, pool=None):
    N, H, L = V.shape[:-1]
    Vdummy = torch.ones((N, H, L, 1), device=V.device)
    
    if not (pool is None):
        product, normalization = pool.starmap(causal_dot_numerator_product, [(Q, K, V, tq, tkv),(Q, K, Vdummy, tq, tkv)])
    else:
        product = causal_dot_numerator_product(Q, K, V, tq, tkv)
        normalization = causal_dot_numerator_product(Q, K, Vdummy, tq, tkv)

    return product / (normalization + 1e-6)

def causal_dot_product_ref(Q, K, V, tq, tkv):
    #product_ref = causal_dot_numerator_product_ref(Q, K, V)

    N, H, L = V.shape[:-1]
    
    Vdummy = torch.ones((N, H, L, 1), device=V.device)

    with Pool(2) as pool:
        product_ref, normalization_ref = pool.starmap(causal_dot_numerator_product_ref, [(Q, K, V),(Q, K, Vdummy)])

    ref_output = product_ref / (normalization_ref+1e-6)
    return ref_output


class CausalDotProductNumerator(torch.autograd.Function):
    """Compute the weighted sum of values but attending only to previous
    values."""
    dot_numerator = {
        "cpu": causal_dot_numerator_product_cpu,
        "cuda": causal_dot_numerator_product_cuda
    }
    dot_numerator_backward = {
        "cpu": causal_dot_numerator_backward_cpu,
        "cuda": causal_dot_numerator_backward_cuda
    }

    @staticmethod
    def forward(ctx, Q, K, V, tq, tkv):
        # Save the inputs for the gradient computation
        ctx.save_for_backward(Q, K, V, tq, tkv)

        # Create the output tensor
        device = Q.device
        N, H, L, _ = Q.shape
        _, _, _, M = V.shape
        
        product = torch.zeros((N, H, L, M), device=device)

        # Actually perform the numerator of dot product
        CausalDotProductNumerator.dot_numerator[device.type](
            Q.data,
            K.data,
            V.data,
            tq, tkv,
            product
        )

        return product

    @staticmethod
    def backward(ctx, grad_out):
        # Extract the saved tensors
        Q, K, V, tq, tkv = ctx.saved_tensors

        # Allocate memory for the gradients
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)
        # Actually compute the gradients
        CausalDotProductNumerator.dot_numerator_backward[Q.device.type](
            Q.data,
            K.data,
            V.data,
            tq, tkv,
            grad_out,
            grad_Q,
            grad_K,
            grad_V
        )

        return grad_Q, grad_K, grad_V, None, None


class CausalDotProductRef(torch.autograd.Function):
    """Compute the weighted sum of values but attending only to previous
    values."""
    dot_numerator = {
        "cpu": causal_dot_product_cpu,
        "cuda": causal_dot_product_cuda
    }
    dot_numerator_backward = {
        "cpu": causal_dot_backward_cpu,
        "cuda": causal_dot_backward_cuda
    }

    @staticmethod
    def forward(ctx, Q, K, V):
        # Save the inputs for the gradient computation
        ctx.save_for_backward(Q, K, V)

        # Create the output tensor
        device = Q.device

        N, H, L, _ = Q.shape
        _, _, _, M = V.shape
        product = torch.zeros((N, H, L, M), device=device)

        # Actually perform the numerator of dot product
        CausalDotProductRef.dot_numerator[device.type](
            Q.data,
            K.data,
            V.data,
            product
        )

        return product

    @staticmethod
    def backward(ctx, grad_out):
        # Extract the saved tensors
        Q, K, V = ctx.saved_tensors

        # Allocate memory for the gradients
        grad_Q = torch.zeros_like(Q)
        grad_K = torch.zeros_like(K)
        grad_V = torch.zeros_like(V)

        # Actually compute the gradients
        CausalDotProductRef.dot_numerator_backward[Q.device.type](
            Q.data,
            K.data,
            V.data,
            grad_out,
            grad_Q,
            grad_K,
            grad_V
        )
        
        return grad_Q, grad_K, grad_V


# Alias the autograd functions to python style snake case naming
causal_dot_numerator_product = CausalDotProductNumerator.apply
causal_dot_numerator_product_ref = CausalDotProductRef.apply
