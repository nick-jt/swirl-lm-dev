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
"""Communication library for the lagrangian particles."""

from typing import Sequence, TypeAlias
import numpy as np
import itertools
from swirl_lm.communication import send_recv
from swirl_lm.physics.lpt import lpt_types
from swirl_lm.physics.lpt import lpt_utils
from swirl_lm.utility import common_ops
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldMap: TypeAlias = types.FlowFieldMap


def pairwise(
    locs: tf.Tensor,
    states: FlowFieldMap,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    variables: Sequence[str],
    grid_spacings: tf.Tensor,
    core_spacings: tf.Tensor,
    local_min_pt: tf.Tensor,
    global_min_pt: tf.Tensor,
    n_max: int,
) -> tf.Tensor:
  """Requests fluid data for particles located on other replicas.

  This function conducts the an n^2 number of exchanges for the `n` replicas.
  In other words, each replica sends and receives fluid data from all other
  replicas. To do this, it iterates through a for loop over the number of
  replicas, each iteration constructing a new set of source-destination pairs.
  The source-destination pairs are constructed so that replica `a` will be
  sending data to replica `b` and replica `b` will be sending data to replica
  `a` for all `a` and `b` in the range `[0, n)`. When the the source and
  destination are the same, the send_recv is called anyways.

  Args:
    locs: An (n, 3) floating point tensor of `n` physical locations that the
      current replica is requesting fluid data at in z, x, y order.
    states: A dictionary containing the fluid data.
    replica_id: The replica id of the local replica.
    replicas: A numpy array of shape `(cx, cy, cz)` containing the replica id of
      each core.
    variables: A sequence of variables to be exchanged (e.g., ["w", "u", "v"]).
      These should exist in `states`.
    grid_spacings: A tensor of the grid spacings in order z, x, y. If a
      non-uniform grid is used, these grid spacings correspond to the values in
      the mapped space which should be a tensor of ones.
    core_spacings: A tensor of the core's partial domain size in the z, x, y
      directions.
    local_min_pt: The minimum point of the local grid in z, x, y including
      halos. This should correspond to the location of the grid point located at
      (0, 0, 0) in the states tensors.
    global_min_pt: The global minimum point of the simulation domain in order z,
      x, y.
    n_max: The maximum number of elements that any given replica can send to
      another replica. This value should be assigned to be at a maximum the
      number of rows in the `lpt_field_` tensors.

  Returns:
    A tensor of size `(n, len(variables))` containing the fluid data requested
    from remote locations.
  """
  loc_replica_ids = lpt_utils.get_particle_replica_id(
      locs, core_spacings, replicas, global_min_pt
  )

  fluid_data = tf.zeros(
      (tf.shape(locs)[0], len(variables)), dtype=lpt_types.LPT_FLOAT
  )

  # Iterate through each circular exchange.
  num_replicas = replicas.size
  source = np.arange(num_replicas, dtype=lpt_types.LPT_NP_INT)
  for offset in range(num_replicas):

    # The source-destination pairs for the current exchange.
    dest = np.roll(source, offset, axis=0).astype(lpt_types.LPT_NP_INT)[::-1]
    source_dest_pair = np.stack([source, dest], axis=1)

    with tf.name_scope("gathering_particle_data"):
      # Determine this replica's destination in the current exchange.
      dest_replica = tf.gather(dest, replica_id)

      # Gathering the locations of the particles owned by this replica but
      # located on the dest replica.
      dest_replica_loc_indices = tf.reshape(
          tf.where(loc_replica_ids == dest_replica), [-1]
      )
      dest_replica_loc = (
          tf.one_hot(dest_replica_loc_indices,
                     tf.shape(locs)[0]) @ locs
      )

    with tf.name_scope("send_recv_locations"):
      # Sending the particle locations to the dest replica to conduct
      # interpolation of fluid data, receiving locations owned by them but
      # located on this replica.
      dest_locations = send_recv.send_recv(
          dest_replica_loc, source_dest_pair, n_max
      )

    # TODO(ntricard): Figure out why when n_max > tf.shape(locs)[0] the
    # fluid_data ends up returning a tensor where each component equals the
    # two components summed, while when n_max = tf.shape(locs)[0]
    # (as we do here) it works.

    # Interpolating at dest locations.
    with tf.name_scope("interpolating_fluid_data"):
      fluid_data_at_dest_locations = lpt_utils.fluid_data_linear_interpolation(
          dest_locations, states, variables, grid_spacings, local_min_pt
      )

    with tf.name_scope("send_recv_fluid_data"):
      # Sending interpolated fluid data to the replica while receiving fluid
      # data from the replica.
      replica_fluid_data = send_recv.send_recv(
          fluid_data_at_dest_locations, source_dest_pair, n_max
      )

    with tf.name_scope("update_fluid_data"):
      fluid_data = lpt_utils.tensor_scatter_update(
          fluid_data, dest_replica_loc_indices, replica_fluid_data
      )

  return fluid_data


def one_shuffle(
    locs: tf.Tensor,
    states: FlowFieldMap,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
    variables: Sequence[str],
    grid_spacings: tf.Tensor,
    core_spacings: tf.Tensor,
    local_min_pt: tf.Tensor,
    global_min_pt: tf.Tensor,
    n_max: int,
) -> tf.Tensor:
  """Circular replica communication to get particle fluid data.

  This function conducts a single circular exchange for the `n` replicas.
  Each replica sends data to the next replica in the circle and receives data
  from the previous replica in the circle. Every iteration, each replica
  interpolates the incoming particle locations that are physically located
  within that replicas domain, and inserts the data into the circulating data.
  After this process is repeated `n` times, each replica will receive the fluid
  data for all of its particle locations.

  Args:
    locs: An (n, 3) floating point tensor of `n` physical locations that the
      current replica is requesting fluid data at in z, x, y order.
    states: A `FlowFieldMap` containing the fluid data.
    replica_id: The replica id of the local replica.
    replicas: A numpy array of shape `(cx, cy, cz)` containing the replica id of
      each core.
    variables: A sequence of variables to be exchanged (e.g., ["w", "u", "v"]).
      These should exist in `states`.
    grid_spacings: A tensor of the grid spacings in order z, x, y. If a
      non-uniform grid is used, these grid spacings correspond to the values in
      the mapped space which should be a tensor of ones.
    core_spacings: A tensor of the core's partial domain size in the z, x, y
      directions.
    local_min_pt: The minimum point of the local grid in z, x, y including
      halos. This should correspond to the location of the grid point located at
      (0, 0, 0) in the states tensors.
    global_min_pt: The global minimum point of the simulation domain in order z,
      x, y excluding halos.
    n_max: The maximum number of elements that any given replica can send to
      another replica. This value should be assigned to be the number of rows in
      the `lpt_field_` tensors.

  Returns:
    A tensor of size `(n, len(variables))` containing the fluid data requested
    from remote locations.
  """
  # Construct the circular exchange.
  num_replicas = replicas.size
  source = np.arange(num_replicas, dtype=lpt_types.LPT_NP_INT)
  dest = np.roll(source, 1, axis=0).astype(lpt_types.LPT_NP_INT)
  source_dest_pairs = np.stack([source, dest], axis=1)

  # Preparing joint location and field data tensor.
  fluid_data = tf.zeros((n_max, len(variables)), dtype=tf.float32)
  loc_and_fluid_data = tf.concat([locs, fluid_data], axis=1)

  for _ in range(num_replicas):

    with tf.name_scope("exchange_fluid_data"):
      # Processing the exchange with the neighbors in the circle.
      loc_and_fluid_data = tf.raw_ops.CollectivePermute(
          input=loc_and_fluid_data, source_target_pairs=source_dest_pairs
      )

    with tf.name_scope("extract_fluid_data"):
      locs = loc_and_fluid_data[:, :3]
      fluid_data = loc_and_fluid_data[:, 3:]

      # Gathering particles that are located on this replica.
      loc_replicas = lpt_utils.get_particle_replica_id(
          locs, core_spacings, replicas, global_min_pt
      )
      loc_indices_local = tf.reshape(tf.where(loc_replicas == replica_id), [-1])
      locs_local = tf.einsum(
          "qj,ji->qi", tf.one_hot(loc_indices_local, n_max), locs
      )

    # Interpolating at dest locations.
    with tf.name_scope("interpolating_fluid_data"):
      fluid_data_at_locs = lpt_utils.fluid_data_linear_interpolation(
          locs_local, states, variables, grid_spacings, local_min_pt
      )

    with tf.name_scope("update_fluid_data"):
      fluid_data = lpt_utils.tensor_scatter_update(
          fluid_data, loc_indices_local, fluid_data_at_locs
      )

      # Preparing joint location-field data tensor.
      loc_and_fluid_data = tf.concat([locs, fluid_data], axis=1)

  return loc_and_fluid_data[:, 3:]


def neighbor_exchange(
    lpt_field_ints: lpt_types.LptFieldInts,
    lpt_field_floats: lpt_types.LptFieldFloats,
    dest_replicas: tf.Tensor,
    replica_id: tf.Tensor,
    replicas: np.ndarray,
) -> tuple[lpt_types.LptFieldInts, lpt_types.LptFieldFloats]:
  """A direct neighbor-wise exchange of particles.

  This method is used in the particle-exchange communication approach for
  Lagrangian particle tracking. Here, we iterate through all neighboring cores
  and conduct communications with those cores, sending particles that
  leave our core to enter the core we are sending to and receiving particles
  from cores we are receiving from in that iteration.

  Args:
    lpt_field_ints: A tensor containing integer field values for the particles.
    lpt_field_floats: A tensor with floating point values for the particles.
    dest_replicas: A 1-D tensor of size num particles that defines their current
      replica location. Can be obtained from `get_particle_replica_id`.
    replica_id: The ID of the calling replica.
    replicas: A 3-D numpy array defining all replicas in the domain.

  Returns:

  """
  # Initializing receiving tensors.
  new_particle_ints = tf.zeros_like(lpt_field_ints)
  new_particle_floats = tf.zeros_like(lpt_field_floats)

  # Gathering computation shape
  cx, cy, cz = replicas.shape

  # Accumulating received particles.
  recv_part_count = tf.constant(0, lpt_types.LPT_INT)

  # The array `replicas` is used here to find the src and dest in the exchange.
  # We iterate over all 6 sides around our core to determine which side to send
  # to: `(+x, -x, +y, -y, +z, -z)`. Simultaneously, we determine cores we are
  # receiving from (-x, +x, -y, +y, -z, +z). For the below lines, when
  # `dir_idx = 0`, we are referring to the  `+x` direction, which provides
  # `p, q, w = (1, 0, 0)`. Likewise, when `dir_idx = 3`, it is the `-y`
  # direction, and `p, q, w = (0, -1, 0)`. For periodic boundaries, the modulus
  # ensures we send from top to bottom and from bottom to top.
  for dir_idx, sign in itertools.product(range(3), (1, -1)):
    with tf.name_scope("creating_send_recv_tensor"):
      p, q, w = tuple(np.roll([1, 0, 0], dir_idx) * sign)
      source_dest_pairs = np.array(
          [
              [
                  replicas[i, j, k], replicas[(i + p) % cx, (j + q) % cy,
                                              (k + w) % cz]
              ]
              for i, j, k in itertools.product(range(cx), range(cy), range(cz))
          ]
      )
      send_to = tf.convert_to_tensor(source_dest_pairs)[replica_id, 1]

    with tf.name_scope("determining_which_particles_to_send"):
      send_particle_indices = tf.where(tf.equal(dest_replicas, send_to))
      send_particle_ints = tf.gather_nd(lpt_field_ints, send_particle_indices)
      send_particle_floats = tf.gather_nd(
          lpt_field_floats, send_particle_indices
      )

    # TODO(ntricard): consider meshing floats and ints into one collective send.
    with tf.name_scope("send_recv_ints"):
      recv_particle_floats = send_recv.send_recv(
          send_particle_floats, source_dest_pairs, lpt_field_ints.shape[0]
      )
    with tf.name_scope("send_recv_floats"):
      recv_particle_ints = send_recv.send_recv(
          send_particle_ints, source_dest_pairs, lpt_field_ints.shape[0]
      )

    with tf.name_scope("extracting_new_particles_to_add"):
      bool_mask = tf.cast(lpt_field_ints[:, 1], tf.bool)
      recv_particle_ints = tf.boolean_mask(recv_particle_ints, bool_mask)
      recv_particle_floats = tf.boolean_mask(recv_particle_floats, bool_mask)

    with tf.name_scope("adding_new_particle_to_stack"):
      num_new_particles = tf.reduce_sum(lpt_field_ints[:, 1])
      indices = tf.range(tf.shape(lpt_field_ints)[0]) + recv_part_count
      new_particle_ints = tf.tensor_scatter_nd_update(
          new_particle_ints, indices[:num_new_particles, None],
          recv_particle_ints
      )
      new_particle_floats = tf.tensor_scatter_nd_update(
          new_particle_floats, indices[:num_new_particles, None],
          recv_particle_floats
      )
      recv_part_count += num_new_particles

  # Adjusting sizes to be just new particles.
  with tf.name_scope("removing_empty_space"):
    indices = tf.range(tf.shape(lpt_field_ints)[0])
    new_particle_ints = tf.gather(new_particle_ints, indices[:recv_part_count])
    new_particle_floats = tf.gather(
        new_particle_floats, indices[:recv_part_count]
    )

  return new_particle_ints, new_particle_floats
