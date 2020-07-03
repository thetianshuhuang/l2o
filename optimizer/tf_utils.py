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


def _var_key(var):
    """Key for representing a primary variable, for looking up slots.
    In graph mode the name is derived from the var shared name.
    In eager mode the name is derived from the var unique id.
    If distribution strategy exists, get the primary variable first.
    Args:
        var: the variable.
    Returns:
        the unique name of the variable.
    """

    # pylint: disable=protected-access
    # Get the distributed variable if it exists.
    if hasattr(var, "_distributed_container"):
        var = var._distributed_container()
    if var._in_graph_mode:
        return var._shared_name
    return var._unique_id
