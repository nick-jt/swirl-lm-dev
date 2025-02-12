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
  dvdt_rhs = -c_d / tau_p * (v_p - v_f) + f_g,
  ```

  where `c_d` is the drag coefficient, `tau_p` is the relaxation time, and `v_f`
  is the fluid velocity at the location of the particle. The parameters `c_d`
  and `tau_p` are constants and are defined in initializing this class.
  Additionally, `F_g` is the force of gravity. This model is not the most
  physically accurate but is kept for historical reasons.

haider_levenspiel:

  This model is based on the Haider-Levenspiel drag model [1].

  ```
  dvdt_rhs = f_d * (v_f - v_p) + f_g,
  f_d = 18 * mu / (rho_p * d_p^2) * c_d * Re / 24,
  Re = |v_f - v_p| * d_p / nu,
  c_d = 24 / (Re + eps) * (1 + b1 * Re^b2) + (b3 * Re) / (b4 + Re),
  ```

  where `f_d` is the drag force, `v_f` is the fluid velocity at the location of
  the particles, `mu` is the dynamic viscosity, `rho_p` is the particle density,
  `d_p` is the particle diameter, `nu` is the kinematic viscosity, `Re` is the
  Reynolds number, `c_d` is the drag coefficient, `eps` is a small number to
  avoid division by zero, and `b1`, `b2`, `b3`, and `b4` are evaluated using
  `phi` using equations found in the original paper. This model has been found
  to be experimentally valid for ember modeling [2] because it accounts for
  non-sphericity, phi, defined as the ratio of the equivalent spherical surface
  area over the actual surface area of the particle.

  [1] Haider and Levenspiel, Powder Technology, 58:63-70, 1989.
  [2] Wadhwani et al., Int. J. of Wildland Fire 31(6) 634-648, 2023.
"""

import numpy as np
from swirl_lm.equations import common
from swirl_lm.physics import constants
import tensorflow as tf


def required_fluid_vars(self):
  """Returns the fluid variables required by the LPT model."""

  required_vars = ["w", "u", "v"]

  if self.params.lpt.WhichOneof("drag_model") == "haider_levenspiel":
    required_vars.append(common.KEY_RHO)

  return required_vars


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
  elif rhs_func == "haider_levenspiel":
    dvdt = dvdt_haider_levenspiel_rhs(
        self, part_locs, part_vels, part_masses, **kwargs
    )
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

  fluid_speeds = kwargs.get("fluid_vars", None)

  c_d = self.params.lpt.manual_drag.c_d
  tau_p = self.params.lpt.manual_drag.tau_p

  dvdt = (
      c_d / tau_p * (fluid_speeds - part_vels) +
      tf.constant(self.gravity_direction) * constants.G
  )

  return dvdt


def dvdt_haider_levenspiel_rhs(
    self, part_locs, part_vels, part_masses, **kwargs
):
  del part_locs, part_masses

  fluid_vars = kwargs.get("fluid_vars", None)
  fluid_speeds = fluid_vars[:, :3]
  fluid_densities = fluid_vars[:, 3]

  # Calculate the Reynolds number based on diameter of a sphere of equivalent
  # volume.
  d_p = self.params.lpt.haider_levenspiel.d_p
  nu = self.params.nu
  mu = nu * fluid_densities
  re = get_reynolds_number(part_vels, fluid_speeds, d_p, nu)

  # Calculate the drag coefficient based on the Haider-Levenspiel model.
  eps = 1e-10
  phi = self.params.lpt.haider_levenspiel.phi
  b1 = tf.math.exp(2.3288 - 6.4581 * phi + 2.4486 * phi**2)
  b2 = 0.0964 + 0.5565 * phi
  b3 = tf.math.exp(4.905 - 13.8944 * phi + 18.4222 * phi**2 - 10.2599 * phi**3)
  b4 = tf.math.exp(1.4681 + 12.2584 * phi - 20.7322 * phi**2 + 15.8855 * phi**3)
  c_d = 24 / (re + eps) * (1 + b1 * re**b2) + (b3 * re) / (b4 + re)

  # Use calculated drag coefficient for dvdt equation.
  rho_p = self.params.lpt.haider_levenspiel.rho_p
  dvdt = dvdt_generic_second_order_eq(
      part_vels, fluid_speeds, c_d, re, mu, rho_p, d_p
  )

  # Account for gravitational acceleration.
  dvdt += tf.constant(self.gravity_direction) * constants.G

  return dvdt


def dvdt_generic_second_order_eq(v_p, v_f, c_d, reynolds_no, mu, rho_p, d_p):
  """A generic second order equation to calculate dvdt given c_d."""
  f_d = 18 * mu / (rho_p * d_p**2) * c_d * reynolds_no / 24
  return tf.einsum('i,ij->ij', f_d, (v_f - v_p))


def get_reynolds_number(v_p, v_f, d_p, nu):
  """Calculates the Reynolds number for the particle."""
  return tf.norm(v_p - v_f) * d_p / nu
