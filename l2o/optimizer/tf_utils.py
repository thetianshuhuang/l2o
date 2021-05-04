"""Private helper functions copied from tensorflow source code."""

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Excerpt from ``tensorflow/python/keras/optimizer_v2/optimizer_v2.py``
# Modified to be compatible with tf 2.3 and 2.4

from tensorflow.python.distribute import distribute_utils
from tensorflow.python.framework import ops


def _var_key(var):
    """Returns slot key for `var`."""
    # pylint: disable=protected-access
    if hasattr(var, "_distributed_container"):
        if callable(var._distributed_container):
            var = var._distributed_container()
        else:
            var = var._distributed_container
    if (distribute_utils.is_distributed_variable(var)
            and not ops.executing_eagerly_outside_functions()):
        return (var.graph, var._shared_name)
    if hasattr(var, "op"):
        return (var.op.graph, var.op.name)
    return var._unique_id
    # pylint: enable=protected-access
