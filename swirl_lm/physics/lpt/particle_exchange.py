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
"""Library for the particle exchange communication scheme.

In this approach, particles are stored and controlled by the device whos partial
domain they fall within and are exchanged between devices as they cross over
from one device's partial domain to another.
"""

from typing import TypeAlias
import numpy as np
from swirl_lm.physics.lpt import lpt
from swirl_lm.physics.lpt import lpt_comm
from swirl_lm.physics.lpt import lpt_types
from swirl_lm.physics.lpt import lpt_utils
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldMap: TypeAlias = types.FlowFieldMap

FIELD_VALS = ["w", "u", "v"]

LPT_INTS_KEY = lpt_types.LPT_INTS_KEY
LPT_FLOATS_KEY = lpt_types.LPT_FLOATS_KEY
LPT_COUNTER_KEY = lpt_types.LPT_COUNTER_KEY


class ParticleExchange(lpt.LPT):
  """Class for the particle exchange communication scheme."""

  def update_particles(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Updates the particle trajectories and locations.

    Args:
      replica_id: The ID of the replica that is calling this function.
      replicas: A 3D numpy array containing the replica ID with the global
        domain.
      states: The fluid state fields. See `swirl_lm.base.parameters` for more
        details.
      additional_states: A dictionary of additional particle and fluid states.
        This includes the Lagrangian particle (lpt) states.

    Returns:
      A `FlowFieldMap` containing the updated particle states.
    """
    particles_generated_per_replica = additional_states[LPT_COUNTER_KEY]

    lpt_field_ints = additional_states[LPT_INTS_KEY]
    lpt_field_floats = additional_states[LPT_FLOATS_KEY]

    # Extracting active particle data.
    with tf.name_scope("extracting_active_particles"):
      bool_mask = tf.cast(lpt_field_ints[:, 1], tf.bool)
      lpt_floats_active = tf.boolean_mask(lpt_field_floats, bool_mask)
      lpt_ints_active = tf.boolean_mask(lpt_field_ints, bool_mask)
      locs_local = lpt_floats_active[:, :3]

    # Gathering fluid properites at the particle locations.
    with tf.name_scope("interpolate_fluid_data"):
      local_min_pt = self._get_local_min_loc(replicas, replica_id)
      fluid_vels = lpt_utils.fluid_data_linear_interpolation(
          locs_local, states, FIELD_VALS, self.grid_spacings_zxy, local_min_pt
      )

    # TODO(ntricard): Add mass consumption rate function.
    omegas = tf.zeros_like(lpt_floats_active[:, 0], dtype=lpt_types.LPT_FLOAT)

    # Time step the particles, updating their attributes.
    with tf.name_scope("time_step_particles"):
      lpt_ints_active, lpt_floats_active = self.increment_time(
          replica_id,
          replicas,
          lpt_ints_active,
          lpt_floats_active,
          additional_states,
          fluid_vels,
          omegas,
      )
      new_locs = lpt_floats_active[:, :3]

    # Modulus the locations across periodic boundaries.
    with tf.name_scope("apply_periodic_boundary_conditions"):
      lpt_field_floats = self._apply_periodic_boundary_conditions(
          lpt_field_floats
      )

    # Removing particles that have exited the domain or have vaporized.
    with tf.name_scope("removing_exiting_particles"):
      dest_replicas = lpt_utils.get_particle_replica_id(
          new_locs, self.core_spacings, replicas, self.global_min_pt
      )
      lpt_ints_active = self._remove_particles(
          lpt_ints_active, lpt_floats_active, dest_replicas
      )

    # Sending and receiving particles.
    with tf.name_scope("sending_receiving_particles"):
      new_lpt_ints, new_lpt_floats = lpt_comm.neighbor_exchange(
          lpt_field_ints, lpt_field_floats, dest_replicas, replica_id, replicas
      )

    with tf.name_scope("adding_particles_to_list"):
      lpt_ints_active, lpt_floats_active = self._add_new_particles(
          lpt_ints_active, lpt_floats_active, new_lpt_ints, new_lpt_floats
      )

    # Replace the field tensors with the new ones.
    lpt_field_ints = tf.ensure_shape(lpt_ints_active, lpt_field_ints.shape)
    lpt_field_floats = tf.ensure_shape(
        lpt_floats_active, lpt_field_floats.shape
    )

    return {
        LPT_INTS_KEY: lpt_field_ints,
        LPT_FLOATS_KEY: lpt_field_floats,
        LPT_COUNTER_KEY: particles_generated_per_replica,
    }
