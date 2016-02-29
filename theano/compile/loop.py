import numpy

import theano
from theano import gof, tensor
from theano.compile import SharedVariable, rebuild_collect_shared
from theano.compile.ops import specify_shape
from theano.compile.function_module import orig_function

from theano.tensor import opt
from theano.tensor.opt import ShapeFeature


class PastValueOp(gof.Op):
    __props__ = ('init_values', 'inplace')
    def __init__(self, init_values, inplace=False):
        assert isinstance(init_values, tuple)
        self.init_values = init_values
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, inp):
        type = inp.type
        self.values = [type.filter_value(v) for v in self.init_values]
        node = Apply(self, [inp], [inp.type()])


class LoopItem(object):
    """Represents one iteration buffer (input or output)

    outer corresponds to the input/output value in the outer graph, may be None

    inner is the variable in the inner graph which matches
    """
    def __init__(self, outer, inner):
        self.outer = outer
        self.inner = inner


class LoopBase(gof.Op):
    """Base class for loop operations

    This class contains shared implementation and state for all Loop
    operations (except Scan). You should never use it directly in a
    graph.
    """
    def __init__(self, inputs, outputs, others,
                 input_hints=None, output_hints=None):
        if others is None:
            others = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        if not isinstance(inputs, list):
            inputs = [inputs]

        for i in inputs + outputs + others:
            if not isinstance(i, gof.Variable):
                raise TypeError(
                    'inputs, outputs and others must be Variables', i)

        if any(isinstance(i, SharedVariable) for i in inputs):
            raise ValueError("Inputs can't be shared variables "
                             "for the inner graph")

        if any(isinstance(o, SharedVariable) for o in others):
            raise ValueError("Don't pass shared variables in others, "
                             "they will be handled automatically")
        if any(isinstance(o, gof.Constant) for o in others):
            raise ValueError("Don't pass constants in others, "
                             "they will be handled automatically")

        self.inputs = inputs
        self.outputs = outputs
        self.others = others

        # To correctly support shared variables the inner function
        # must not see them. Otherwise it becomes impossible to
        # compute the gradient.  So we collect them here to hide them
        # later.
        self.shared = [var for var in gof.graph.inputs(outputs, inputs+others)
                       if isinstance(var, SharedVariable)]

        if input_hints is None:
            self.input_hints = [inp.type.clone(
                    broadcastable(False,) + inp.type.broadcastable)()
                           for inp in inputs]
        else:
            # We don't want to reuse the passed-in variable, just its type
            self.input_hints = [inp.clone() for inp in input_hints]
            assert len(self.input_hints) == len(self.inputs)

        if output_hints is None:
            self.output_hints = [out.type.clone(
                    broadcastable=(False,) + out.type.broadcastable)()
                                 for out in outputs]
        else:
            # We don't want to reuse the passed-in variable, just its type
            self.output_hints = [out.clone() for out in output_hints]
            assert len(self.output_hints) == len(self.outputs)

    def __eq__(self, other):
        #TODO: recognize a copy
        return self is other

    def __hash__(self):
        #TODO: use internal variables in hash
        return hash(type(self))

    def make_func_g(self, init_i):
        i = theano.shared(numpy.asarray(init_i, dtype='uint64'))
        shared_g = [var.type() for var in self.shared]
        inputs_g = [inp[i] for inp in self.input_hints]
        outputs_g = [theano.tensor.set_subtensor(out_h[i], out)
                     for out_h, out in zip(self.output_hints, self.outputs)]

        new = rebuild_collect_shared(outputs_g,
                                     inputs=(self.inputs + self.others +
                                             self.output_models +
                                             self.shared),
                                     replace=dict(zip(self.shared +
                                                      self.inputs,
                                                      shared_g + inputs_g)),
                                     copy_inputs_over=False,
                                     rebuild_strict=True)
        (new_inputs, new_outputs,
         [clone_d, update_d, update_expr, shared_inputs]) = new
        assert len(new_inputs) == (len(self.inputs) + len(self.others) +
                                   len(self.output_models) + len(self.shared))
        assert len(new_outputs) == len(self.outputs)
        assert shared_inputs == [i]

        return i, new_inputs, new_outputs

    def make_shape_graph(self, input_shapes):
        # Here input_shapes doesn't contain the shapes of the output
        # models, that would be silly.
        allinps = self.inputs + self.others
        assert len(input_shapes) == len(allinps)

        # This is a bit of a hackish usage of ShapeFeature, but it
        # simplifies things immensely compared to going though a real
        # FunctionGraph
        sf = ShapeFeature()
        sf.on_attach(gof.FunctionGraph([], []))

        for i, i_s in zip(allinps, input_shapes):
            sf.set_shape(i, i_s)

        def local_traverse(out):
            if out in sf.shape_of:
                return
            elif out.owner is None:
                sf.init_r(out)
            else:
                for i in out.owner.inputs:
                    if not i in sf.shape_of:
                        local_traverse(i)
                # ShapeFeature does not actually use the fgraph
                sf.on_import(None, out.owner, reason='')
        ret = []
        for o in self.outputs:
            local_traverse(o)
            ret.append(sf.shape_of[o])
        # XXX Maybe clone return nodes?
        return ret

    def prepare_node(self, node, storage_map, compute_map, no_recycling):
        # XXX: maybe do some hocus pocus to share storage_map
        # Although this wouldn't be safe to share for more than one
        # graph we would just have to return a unique thunk from here.
        if not hasattr(self, "fn"):
            self.fn, self._i, _, _ = self.make_func()

    def infer_shape(self, inputs, inputs_shapes):
        os = input_shapes[len(self.inputs_hints) + len(self.others):]
        return os[:len(self.output_hints)]

    def perform(self, node, inputs, outputs):
        self._i.set_value(0)
        for c, v in zip(self.fn.inputs, inputs[1:]):
            c.storage[0] = v
        self.fn.fn(n_calls=inputs[0])
        assert len(self.fn.outputs) == len(outputs)
        for o, c in zip(outputs, self.fn.outputs):
            o[0] = c.storage[0]


class Loop(LoopBase):
    """This creates a loop from inputs and outputs lists of variables.

    :param inputs: list of inputs to loop over

    :param outputs: list of output expressions

    :param others: other variables that will be used to compute outputs.
        Shared variables and constants must not be part of this list.
    """

    def make_node(self, n_iters, *vars):
        # Check that the number of iterations is a scalar
        assert n_iters.ndim == 0
        assert n_iters.type.dtype == 'int64'

        # First in vars is all the inputs which are iterated
        inputs = vars[:len(self.input_hints)]
        if len(inputs) != len(self.input_hints):
            raise ValueError("Not enough inputs")
        for oi, ii in zip(inputs, self.input_hints):
            if not oi.type == ii.type:
                raise TypeError("Wrong type for input, expected %s but got %s"
                                % (ii.type, oi.type))
        vars = vars[len(self.input_hints):]
        # After that is the others
        others = vars[:len(self.others)]
        if len(others) != len(self.others):
            raise ValueError("Not enough others")
        for oi, ii in zip(others, self.others):
            if not oi.type == ii.type:
                raise TypeError("Wrong type for other, expected %s but got %s"
                                % (ii.type, oi.type))

        # Finally we have the output buffers
        outputs = vars[len(self.others):]
        if len(outputs) != len(self.output_hints):
            raise ValueError("Not enough outputs")
        for oi, ii in zip(outputs, self.output_hints):
            if not oi.type == ii.type:
                raise TypeError("Wrong type for output, expected %s but got %s"
                                % (ii.type, oi.type))

        return gof.Apply(self,
                         # tackle on the end of our inputs the list of
                         # shared variables we will need for easy
                         # access.
                         [n_iters] + list(vars) + self.shared,
                         [o.clone() for o in self.output_hints])

    def make_func(self):
        i, f_inputs, f_outputs = self.make_func_g(0)

        fn = function(
            f_inputs, f_outputs,
            updates=[(i, i+numpy.asarray(1, dtype='int64'))],
            mode=Mode(linker=VM_Linker(allow_gc=False, use_cloop=True)))

        return fn, i, f_inputs, f_outputs

    def grad(self, inputs, output_grads):
        lopg = LoopGrad(self.inputs, self.outputs, self.others, self.shared,
                        self.input_hints, self.output_hints)
        # no grad for n_steps
        return DisconnectedType()() + lopg(inputs[:1] + output_grads,
                                           return_list=True)

    # # TODO # #
    # def connection_pattern(self, node):


class LoopGrad(LoopBase):
    def make_node(self, n_steps, *inputs):
        # Check that the number of iterations is a scalar
        assert n_iters.ndim == 0
        assert n_iters.type.dtype == 'int64'

        if len(inputs) != len(self.output_hints):
            raise ValueError("Wrong number of inputs")
        for oi, ii in zip(inputs, self.output_hints):
            if not oi.type == ii.type:
                raise TypeError("Wrong type for input, expected %s but got %s"
                                % (ii.type, oi.type))

        outputs = [o.clone() for o in self.input_hints +
                   self.others + self.output_models]
        # We don't want to clone shared variables since we don't want
        # to have SharedVaraibles in the outputs.
        outputs += [var.type() for var in self.shared]

        return gof.Apply(self, [n_iters] + list(inputs), outputs)

    def make_func(self):
        i, f_inputs, f_outputs = self.make_func_g(-1)

        g_inputs = [o.clone() for o in f_outputs]
        g_outputs = theano.grad(None, wrt=f_inputs,
                                known_grads=dict(zip(f_outputs, g_inputs)))

        fn = function(
            g_inputs, g_outputs,
            updates=[(i, i-numpy.asarray(1, dtype='int64'))],
            mode=Mode(linker=VM_Linker(allow_gc=False, use_cloop=True)))

        return fn, i, f_inputs, f_outputs

    # # TODO # #
    # I think this is a regular loop over a double grad of the inner graph
    # def grad(self, inputs, grads):


def loop_fn(n_steps, fn, inputs, others=None, output_hints=None):
    if others is None:
        others = []
    else:
        others = list(others)
    inner_inputs = [i[0] for i in inputs]
    inner_outputs = fn(*(inner_inputs + others))

    lop = Loop(inner_inputs, inner_outputs, others, input_hints=inputs,
               output_hints=output_hints)

    if output_hints is None:
        out_shp = lop.make_shape_graph([[i.shape[j] for j in range(i.ndim)]
                                        for i in inner_inputs + others])
        output_hints = [tensor.zeros((n_steps,) + shp) for shp in out_shp]
        assert all(loi.type == oi.type for lio, io in zip(lop.output_hints,
                                                          output_hints))
    return lop(*((n_steps,) + inputs + others + output_hints))

# Since Loop contains a Theano compiled function, we should let
# DebugMode know about it
gof.ops_with_inner_function[Loop] = 'fn'
gof.ops_with_inner_function[LoopGrad] = 'fn'
