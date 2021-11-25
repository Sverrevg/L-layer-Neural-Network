import kp
import pyshader as ps
import numpy as np

"""
Source: https://kompute.cc/
"""


def quadratic(Z):
    # Create Kompute manager:
    manager = kp.Manager()

    # Initialize tensors:w
    tensor_in_Z = manager.tensor(Z)
    tensor_out_A = manager.tensor(np.zeros(Z.shape))

    sq = manager.sequence()

    sq.eval(kp.OpTensorSyncLocal([tensor_in_Z, tensor_out_A]))

    algo = manager.algorithm([tensor_in_Z, tensor_out_A], compute_shader_quadratic.to_spirv())

    # Run shader operation synchronously:
    sq.eval(kp.OpAlgoDispatch(algo))
    sq.eval(kp.OpTensorSyncLocal([tensor_out_A]))

    return tensor_out_A.data()


@ps.python2shader
def compute_shader_quadratic(index=("input", "GlobalInvocationId", ps.ivec3),
                             data_1=("buffer", 0, ps.Array(ps.f32)),
                             data_2=("buffer", 1, ps.Array(ps.f32))):
    """
    Define code to run on the GPU.
    Now we can add the Kompute algorithm that will be executed on the GPU ('shader' code).
    Format: <param> = ("<memory>", <binding>, <type>, ...)
    Use ps.f32 for float values.
    """

    i = index.x
    data_2[i] = data_1[i] * data_1[i]
