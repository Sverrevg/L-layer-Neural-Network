import kp
import pyshader as ps
import numpy as np

"""
Source: https://kompute.cc/
"""


class math_operations:
    def __init__(self):
        """
        1. Create Kompute manager (selects device 0 by default).
        Kompute manager is in charge of creating and managing underlying Vulkan resources.
        """
        self.manager = kp.Manager()

    def quadratic(self, Z):
        """
        2. Create Kompute Tensors to hold data (1 input, 1 output).
        These will hold the data that will be mapped into the GPU for computing.
        When the tensors are created, the data will have to be mapped into GPU memory.
        """
        tensor_in_Z = self.manager.tensor(Z)
        tensor_out_A = self.manager.tensor(np.zeros(Z.shape))

        """
        3. Initialise the Kompute Tensors in the GPU.
        Now we can map the data into the GPU with the underlying Vulkan buffer.
        """
        self.manager.([tensor_in_Z, tensor_out_A])

        """
        5. Dispatch GPU shader execution against Kompute Tensors.
        """
        self.manager.eval_algo_data_def(
            [tensor_in_Z, tensor_out_A],
            self.compute_shader_quadratic.to_spirv())

        """
        6. Sync tensor data from GPU back to local. 
        Result data is held in GPU memory of the output tensor.
        We can use eval_tensor_sync_local_def to fetch it.
        """
        self.manager.eval_tensor_sync_local_def([tensor_out_A])

        """
        7. Return results.
        """
        return tensor_out_A.data()


@ps.python2shader
def compute_shader_quadratic(index=("input", "GlobalInvocationId", ps.ivec3),
                             data_1=("buffer", 0, ps.Array(ps.f32)),
                             data_2=("buffer", 1, ps.Array(ps.f32))):
    """
    4. Define code to run on the GPU.
    Now we can add the Kompute algorithm that will be executed on the GPU ('shader' code).
    Format: <param> = ("<memory>", <binding>, <type>, ...)
    Use ps.f32 for float values.
    """

    i = index.x  # Fetch the current run index being processed.

    # Perform actual equation:
    data_2[i] = data_1[i] * data_1[i]
