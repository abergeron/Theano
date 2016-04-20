from __future__ import absolute_import, print_function, division
import theano
from theano.tests import unittest_tools as utt

from nose.plugins.skip import SkipTest

from .config import mode_with_gpu, test_ctx_name
from .test_basic_ops import rand_gpuarray

from ..type import GpuArrayType
from ..fft import fft, ifft, skcuda_available

if not skcuda_available:
    raise SkipTest('Optional package skcuda not available')


def test_fft_ifft():
    x = GpuArrayType(broadcastable=[False, False], dtype='float32',
                     context_name=test_ctx_name)()
    cx = fft(x)
    x2 = ifft(cx)

    f = theano.function([x], [cx, x2], mode=mode_with_gpu)

    for sh in (7, 8):
        x_val = rand_gpuarray(2, sh, dtype='float32')

        cx_val, x2_val = f(x_val)

        assert cx_val.ndim == 3
        assert cx_val.shape[-1] == 2

        assert x2_val.shape == x_val.shape

        utt.assert_allclose(x_val, x2_val)
