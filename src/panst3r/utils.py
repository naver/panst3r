# Copyright (C) 2025-present Naver Corporation. All rights reserved.

import torch
import numpy as np
from typing import Callable, Iterable, Optional, Tuple, Union, List
from tqdm import tqdm

def transpose_to_landscape(head, activate=True, dims=(1,2)):
    """ Predict in the correct aspect-ratio,
        then transpose the result in landscape
        and stack everything back together.
    """
    def wrapper_no(decout, true_shape):
        B = len(true_shape)
        assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'
        H, W = true_shape[0].cpu().tolist()
        x = head(decout, (H, W))
        return x

    def nested_compose(l_result, p_result, is_landscape):
        """Combines results of landscape and portrait batches. Works on nested tensors."""

        if isinstance(l_result, dict):
            return {k: nested_compose(l_result[k], p_result[k], is_landscape) for k in l_result}
        elif isinstance(l_result, list):
            return [nested_compose(l, p, is_landscape) for l, p in zip(l_result, p_result)]
        elif isinstance(l_result, tuple):
            return tuple(nested_compose(l, p, is_landscape) for l, p in zip(l_result, p_result))
        elif isinstance(l_result, torch.Tensor):
            B = l_result.size(0) + p_result.size(0)
            res = l_result.new(B, *l_result.shape[1:])
            res[is_landscape] = l_result
            res[~is_landscape] = p_result
            return res

    def wrapper_yes(decout, true_shape):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            return head(decout, (H, W))
        if is_portrait.all():
            return transposed(head(decout, (W, H)), dims)

        # batch is a mix of both portraint & landscape
        selout = lambda ar: [d[ar] for d in decout]
        l_result =            head(selout(is_landscape), (H, W))
        p_result = transposed(head(selout(is_portrait),  (W, H)), dims)

        # allocate full result
        result = nested_compose(l_result, p_result, is_landscape)

        return result

    return wrapper_yes if activate else wrapper_no


def transposed(val, dims=(1,2)):
    hdim, wdim = dims
    if isinstance(val, dict):
        return {k: transposed(v, dims) for k, v in val.items()}
    elif isinstance(val, list):
        return [transposed(v, dims) for v in val]
    elif isinstance(val, tuple):
        return tuple(transposed(v, dims) for v in val)
    elif isinstance(val, torch.Tensor):
        return val.swapaxes(hdim,wdim)


def get_colors_grid(num_colors):
    """Get RGB colors with grid sampling."""
    N = int(np.ceil((num_colors+1) ** (1/3)))

    coords = np.linspace(0, 1, N)
    r,g,b = np.meshgrid(coords, coords, coords)
    colors = np.c_[r.ravel(), g.ravel(), b.ravel()][1:] # Remove black
    np.random.shuffle(colors)

    colors = colors[:num_colors]

    return (colors * 255).astype(np.uint8)

OutType = Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]
def batched_map(
    fn: Callable[..., OutType],
    tensors: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    batch_size: int = None,
    flatten_dims: Optional[Tuple[int, int]] = None,
    split_dim: int = 0,
    multi_ar: bool = False,
    verbose: bool = False,
    desc: Optional[str] = None,
) -> OutType:
    """
    Apply `fn` to aligned mini-batches of `tensors`, with optional pre-flattening.

    Args:
        fn: Callable invoked like `fn(*batch_slices) -> tensor OR (tensors...)`.
        tensors: Input tensor(s). Must match in size along `split_dim` after flattening.
        batch_size: Batch size for splitting. If None all tensors are processed in a single batch.
        flatten_dims: (start_dim, end_dim) to flatten before processing and unflatten at the end; None to skip.
        split_dim: Dimension to split on (after flattening if applied) and concatenate back.
        multi_ar: If True, enables multi-aspect ratio processing -> each tensor is a list of tensors with different aspect ratios.
        verbose: If True, show a tqdm progress bar.
        desc: Optional tqdm description.

    Returns:
        Concatenation of per-batch outputs. If `fn` returns a tensor, returns a tensor.
        If `fn` returns a tuple/list of tensors, returns the same structure with each
        element concatenated along `cat_dim`.
    """

    if isinstance(tensors, torch.Tensor):
        tensors = (tensors,)
    elif multi_ar and isinstance(tensors[0], torch.Tensor):
        tensors = (tensors,)

    if not multi_ar:
        tensors = [[t] for t in tensors] # single aspect ratio

    num_ar = len(tensors[0])
    assert all(len(t) == num_ar for t in tensors), "All tensors must have the same number of multi-ar slices."

    # Shallow copy (prevent editing inputs)
    tensors = [[t for t in tensor_i] for tensor_i in tensors]

    preflatten_shapes = []
    ns = []
    for ar_i in range(num_ar):
        if flatten_dims is not None:
            fstart,fend = flatten_dims
            flt_shapes = [t[ar_i].shape[fstart:fend+1] for t in tensors]
            assert all(sh == flt_shapes[0] for sh in flt_shapes[1:]), "All tensors must have the same shape along flatten dimensions."
            preflatten_shapes.append(flt_shapes[0])

            for tensor in tensors:
                tensor[ar_i] = tensor[ar_i].flatten(fstart, fend)

        split_dims = [t[ar_i].shape[split_dim] for t in tensors]
        n = split_dims[0]
        assert all(sd == n for sd in split_dims[1:]), "All tensors must have the same size along split_dim."
        ns.append(n)

    total_n = sum(ns)
    # Process in chunks
    outs = None
    with tqdm(total=total_n, desc=desc, leave=True, disable=not verbose) as pbar:
        for ar_i in range(num_ar):
            n = ns[ar_i]
            batch_size_i = batch_size if batch_size is not None else n
            starts = range(0, n, batch_size_i)
            outs_i = []

            for start in starts:
                end = min(start + batch_size_i, n)
                span = end - start
                batch = tuple(t[ar_i].narrow(split_dim, start, span) for t in tensors)
                outs_i.append(fn(*batch))
                pbar.update(span)

            def _maybe_unflatten(x: torch.Tensor) -> torch.Tensor:
                if flatten_dims is not None:
                    return x.unflatten(fstart, preflatten_shapes[ar_i])
                return x

            # Concatenate & unflatten outputs
            if isinstance(outs_i[0], torch.Tensor):
                outs_i = torch.cat(outs_i, dim=split_dim)
                outs_i = (_maybe_unflatten(outs_i),)
            elif isinstance(outs_i[0], tuple):
                outs_i = [torch.cat([out[i] for out in outs_i], dim=split_dim) for i in range(len(outs_i[0]))]
                outs_i = tuple([_maybe_unflatten(o) for o in outs_i])
            else:
                raise ValueError("Unsupported output type from fn: {}".format(type(outs_i[0])))

            if outs is None:
                outs = [[cur_t] for cur_t in outs_i]
            else:
                # Append outputs to the respective lists
                for out_list, cur_t in zip(outs, outs_i):
                    out_list.append(cur_t)

    if not multi_ar:
        outs = [l[0] for l in outs]  # single aspect ratio, return single tensors

    # Single output
    if len(outs) == 1:
        outs = outs[0]

    return outs

def unstack_tensors(index_stacks_i, tensor_i):
    num_elements = max([max(index_stack_i) for index_stack_i in index_stacks_i]) + 1
    tensor_unstacked = [None for _ in range(num_elements)]
    for tensor_i_stack, index_stack_i in zip(tensor_i, index_stacks_i):
        for j in range(tensor_i_stack.shape[0]):
            tensor_unstacked[index_stack_i[j]] = tensor_i_stack[j]
    return tensor_unstacked

def get_dtype(amp):
    if amp == "fp16":
        dtype = torch.float16
    elif amp == "bf16":
        assert torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16
    else:
        assert not amp
        dtype = torch.float32
    return dtype