from __future__ import absolute_import, print_function, division

import numpy as np
import theano.tensor as T
from theano import Op, Apply

from .basic_ops import gpu_contiguous, as_gpuarray_variable, infer_context_name
from .type import GpuArrayType

try:
    import pygpu
except ImportError:
    pass

try:
    import skcuda
    from skcuda import fft as skfft
    skcuda.misc.init()
    skcuda_available = True
except (ImportError, Exception):
    skcuda_available = False


class CuFFTOp(Op):
    __props__ = ()

    def __init__(self):
        if not skcuda_available:
            raise RuntimeError("skcuda is required for this Op.")

    def make_node(self, inp):
        ctx_name = infer_context_name(inp)
        inp = as_gpuarray_variable(inp, ctx_name)
        if not inp.type.dtype == 'float32':
            raise ValueError('fft only works on floats')
        inp = gpu_contiguous(inp)
        # This should be of the "complex" dtype, but we instead fake
        # it with a pair of floats
        out = GpuArrayType(broadcastable=[False] * (inp.type.ndim + 1),
                           dtype=inp.type.dtype, context_name=ctx_name)()
        return Apply(self, [inp], [out])

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        plan_input_shape = [None]
        plan = [None]

        def thunk():
            x = inputs[0]
            input_shape = x[0].shape

            # construct output shape
            output_shape = list(input_shape)
            # DFT of real input is symmetric, no need to store
            # redundant coefficients
            output_shape[-1] = output_shape[-1] // 2 + 1
            # extra dimension with length 2 for real/imag
            output_shape += [2]
            output_shape = tuple(output_shape)

            z = outputs[0][0]

            # only allocate if there is no previous allocation of the
            # right size.
            if z[0] is None or z[0].shape != output_shape:
                z[0] = pygpu.zeros(output_shape, dtype='float32',
                                   context=x[0].context)

            with x[0].context:
                # only initialise plan if necessary
                if plan[0] is None or plan_input_shape[0] != input_shape:
                    plan_input_shape[0] = input_shape
                    plan[0] = skfft.Plan(input_shape[1:], np.float32,
                                         np.complex64, batch=input_shape[0])

                skfft.fft(inputs[0][0], outputs[0][0], plan[0])

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk


class CuIFFTOp(Op):
    __props__ = ()

    def __init__(self):
        if not skcuda_available:
            raise RuntimeError("skcuda is required for this Op.")

    def make_node(self, inp):
        ctx_name = infer_context_name(inp)
        inp = as_gpuarray_variable(inp, ctx_name)
        # The input should probably be "complex" but we fake that with
        # two floats.
        if not inp.type.dtype == 'float32':
            raise ValueError('fft only works on floats')
        inp = gpu_contiguous(inp)
        out = GpuArrayType(broadcastable=[False] * (inp.type.ndim - 1),
                           dtype=inp.type.dtype, context_name=ctx_name)()
        return Apply(self, [inp], [out])

    def make_thunk(self, node, storage_map, _, _2):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        plan_input_shape = [None]
        plan = [None]

        def thunk():
            x = inputs[0]
            input_shape = x[0].shape

            # construct output shape
            # chop off the extra length-2 dimension for real/imag
            output_shape = list(input_shape[:-1])
            # restore full signal length
            output_shape[-1] = (output_shape[-1] - 1) * 2
            output_shape = tuple(output_shape)

            z = outputs[0]

            # only allocate if there is no previous allocation of the
            # right size.
            if z[0] is None or z[0].shape != output_shape:
                z[0] = pygpu.zeros(output_shape, dtype='float32',
                                   context=x[0].context)

            with x[0].context:
                # only initialise plan if necessary
                if plan[0] is None or plan_input_shape[0] != input_shape:
                    plan_input_shape[0] = input_shape
                    plan[0] = skfft.Plan(output_shape[1:], np.complex64,
                                         np.float32, batch=output_shape[0])

                skfft.ifft(x[0], z[0], plan[0])
                # strangely enough, enabling rescaling here makes it run
                # very, very slowly.  so do this rescaling manually
                # afterwards!

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

cufft = CuFFTOp()
cuifft = CuIFFTOp()


def fft(val):
    """
    Perform fft on a batch of values.

    This takes a nD values (at least 2D) and peforms an fft on each
    element along the first dimension.

    The return value has one more dimension of size 2 that represents
    the "complex" nature of the output.
    """
    return cufft(val)


def ifft(val):
    """
    Perform ifft on a batch of values.

    This takes a nD values (at least 3D) and peforms an ifft on each
    element along the first dimension.

    The last dimension MUST be of exactly size 2 and represents the
    "complex" nature of the input.

    The output will have one less dimension.
    """
    out = cuifft(val)
    return as_gpuarray_variable((1.0 / T.cast(out.shape[1] * out.shape[2],
                                              'float32')) * out,
                                out.type.context_name)
