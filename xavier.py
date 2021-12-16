
import paddle.fluid.initializer as initializer

from paddle.fluid import framework
from paddle.fluid import core
from paddle.fluid.framework import in_dygraph_mode, default_main_program
import numpy as np
from paddle.fluid.core import VarDesc
from paddle.fluid import unique_name
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype

class XavierInitializerWithGain(initializer.Initializer):
    def __init__(self, uniform=True, fan_in=None, fan_out=None, seed=0, gain=1.0):
        assert uniform is not None
        assert seed is not None
        super(XavierInitializerWithGain, self).__init__()
        self._uniform = uniform
        self._fan_in = fan_in
        self._fan_out = fan_out
        self._seed = seed
        self._gain = gain

    def __call__(self, var, block=None):
        """Initialize the input tensor with Xavier initialization.
        Args:
            var(Tensor): Tensor that needs to be initialized.
            block(Block, optional): The block in which initialization ops
                   should be added. Used in static graph only, default None.
        Returns:
            The initialization op
        """
        block = self._check_block(block)

        assert isinstance(block, framework.Block)
        check_variable_and_dtype(var, "Out",
                                 ["uint16", "float16", "float32", "float64"],
                                 "xavier_init")

        f_in, f_out = self._compute_fans(var)

        # If fan_in and fan_out are passed, use them
        fan_in = f_in if self._fan_in is None else self._fan_in
        fan_out = f_out if self._fan_out is None else self._fan_out

        if self._seed == 0:
            self._seed = block.program.random_seed

        # to be compatible of fp16 initalizers
        if var.dtype == VarDesc.VarType.FP16 or (
                var.dtype == VarDesc.VarType.BF16 and not self._uniform):
            out_dtype = VarDesc.VarType.FP32
            out_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['xavier_init', var.name, 'tmp'])),
                shape=var.shape,
                dtype=out_dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)
        else:
            out_dtype = var.dtype
            out_var = var

        if self._uniform:
            limit = self._gain * np.sqrt(6.0 / float(fan_in + fan_out))
            op = block.append_op(
                type="uniform_random",
                inputs={},
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": out_dtype,
                    "min": -limit,
                    "max": limit,
                    "seed": self._seed
                },
                stop_gradient=True)

        else:
            std = self._gain * np.sqrt(2.0 / float(fan_in + fan_out))
            op = block.append_op(
                type="gaussian_random",
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": out_dtype,
                    "mean": 0.0,
                    "std": std,
                    "seed": self._seed
                },
                stop_gradient=True)

        if var.dtype == VarDesc.VarType.FP16 or (
                var.dtype == VarDesc.VarType.BF16 and not self._uniform):
            block.append_op(
                type="cast",
                inputs={"X": out_var},
                outputs={"Out": var},
                attrs={"in_dtype": out_var.dtype,
                       "out_dtype": var.dtype})

        if not framework.in_dygraph_mode():
            var.op = op
        return 

class XavierUniformWithGain(XavierInitializerWithGain):
    r"""
    This class implements the Xavier weight initializer from the paper
    `Understanding the difficulty of training deep feedforward neural
    networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
    by Xavier Glorot and Yoshua Bengio.
    This initializer is designed to keep the scale of the gradients
    approximately same in all the layers. In case of Uniform distribution,
    the range is [-x, x], where
    .. math::
        x = \sqrt{\\frac{6.0}{fan\_in + fan\_out}}
    Args:
        fan_in (float, optional): fan_in for Xavier initialization, it is
                inferred from the tensor. The default value is None.
        fan_out (float, optional): fan_out for Xavier initialization, it is
                 inferred from the tensor. The default value is None.
        name(str, optional): The default value is None. Normally there is no need for user to set this
            property. For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        A parameter initialized by Xavier weight, using a uniform distribution.
    Examples:
        .. code-block:: python
            import paddle
            data = paddle.ones(shape=[3, 1, 2], dtype='float32')
            weight_attr = paddle.framework.ParamAttr(
                name="linear_weight",
                initializer=paddle.nn.initializer.XavierUniform())
            bias_attr = paddle.framework.ParamAttr(
                name="linear_bias",
                initializer=paddle.nn.initializer.XavierUniform())
            linear = paddle.nn.Linear(2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
            # linear.weight:  [[-0.04229349 -1.1248565 ]
            #                  [-0.10789523 -0.5938053 ]]
            # linear.bias:  [ 1.1983747  -0.40201235]
            res = linear(data)
            # res:  [[[ 1.0481861 -2.1206741]]
            #        [[ 1.0481861 -2.1206741]]
            #        [[ 1.0481861 -2.1206741]]]
    """

    def __init__(self, fan_in=None, fan_out=None, name=None, gain=1.0):
        super(XavierUniformWithGain, self).__init__(
            uniform=True, fan_in=fan_in, fan_out=fan_out, seed=0, gain=gain)
