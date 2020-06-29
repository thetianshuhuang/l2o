# Copyright 2016 Google Inc.
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

# Modified:
# Split factory from networks.py

""""""

import pickle
import sys


from .base import save, Network, StandardDeepLSTM


def factory(net, net_options=(), net_path=None):
    """Network factory."""

    net_class = getattr(sys.modules[__name__], net)
    net_options = dict(net_options)

    if net_path:
        with open(net_path, "rb") as f:
            net_options["initializer"] = pickle.load(f)

    return net_class(**net_options)


__all__ = [
    "factory",
    "save", "Network", "StandardDeepLSTM"
]
