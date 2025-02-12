# Copyright 2024 The swirl_lm Authors.
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
"""Library for the Lagrangian Particle Tracking (LPT) right-hand side equations.

The particles are represented by an ODE system with governing equations:

```
d(x_p) / dt = v_p,
d(v_p) / dt = dvdt_rhs,
d(m_p) / dt = -omega,
```

where `x_p` is particle location, `v_p` is particle velocity, `m_p` is
particle mass, `omega` is the mass consumption rate. For any stretched
dimensions, those dimensions will be tracked in the mapped domain. Each velocity
`v` can be broken down into three components `v0`, `v1`, and `v2`. These
correspond to the `z, x, y` directions in Swirl-LM physical coordinates. The
momentum equation, `dvdt_rhs` depends on the user choice in the proto-file,
with the following models included:

manual_drag:

  ```
  dvdt_rhs = -c_d / tau_p * (v_p - v_f),
  ```

  where `c_d` is the drag coefficient, `tau_p` is the relaxation time, and `v_f`
  is the fluid velocity at the location of the particle. The parameters `c_d`
  and `tau_p` are constants and are defined in initializing this class. This
  model is not the most physically accurate and is kept for historical reasons.

"""

import numpy as np
from swirl_lm.physics import constants
import tensorflow as tf


def particle_rhs(
    self,
    part_locs: tf.Tensor,
    part_vels: tf.Tensor,
    part_masses: tf.Tensor,
    **kwargs,
):
  """Computes the right-hand side of the LPT equations.

  Args:
    part_locs: `n` by 3 tensor of `z, x, y` locations of the `n` particles.
    part_vels: `n` by 3 tensor `z, x, y` velocities of the `n` particles.
    part_masses: `n` sized 1-D tensor of particle masses.
    kwargs: Supplemental Swirl data.

  Returns:
    Evolution terms for particle locations, velocities, and masses.
  """

  # Particle location
  dxdt = dxdt_rhs(self, part_locs, part_vels, part_masses, **kwargs)

  # Particle velocity
  rhs_func = self.params.lpt.WhichOneof("drag_model")
  if rhs_func == "manual_drag":
    dvdt = dvdt_manual_rhs(self, part_locs, part_vels, part_masses, **kwargs)
  else:
    raise NotImplementedError(f"Particle model {rhs_func} not implemented.")

  # Particle Burning
  omegas = kwargs.get("omegas", None)
  dmdt = -omegas

  return dxdt, dvdt, dmdt


def dxdt_rhs(self, part_locs, part_vels, part_masses, **kwargs):
  del part_locs, part_masses
  additional_states = kwargs.get("additional_states", None)
  local_min_loc = kwargs.get("local_min_loc", None)
  # In a stretched grid, dxdt becomes mapped coordinates, while dvdt and
  # dmdt remain in physical domain.
  if np.any(self.use_stretched_grid_zxy):
    # For any non-stretched dimensions, the returned value for
    # `grid_spacings`` is 1.0 because the dxdt equation remains in physical
    # domain for those dimensions.
    with tf.name_scope("lpt_getting_stretched_grid_spacings"):
      grid_spacings = self._get_grid_spacings(additional_states, local_min_loc)
    dxdt = part_vels / grid_spacings
  else:
    dxdt = part_vels

  return dxdt


def dvdt_manual_rhs(self, part_locs, part_vels, part_masses, **kwargs):
  """Computes the right-hand side of the LPT equations."""
  del part_locs, part_masses

  fluid_speeds = kwargs.get("fluid_speeds", None)

  c_d = self.params.lpt.manual_drag.c_d
  tau_p = self.params.lpt.manual_drag.tau_p

  dvdt = (
      c_d / tau_p * (fluid_speeds - part_vels) +
      tf.constant(self.gravity_direction) * constants.G
  )

  return dvdt
