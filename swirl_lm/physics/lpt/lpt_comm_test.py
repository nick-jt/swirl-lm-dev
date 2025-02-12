import itertools

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

from swirl_lm.physics.lpt import lpt_comm


class NeighborExchangeTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(NeighborExchangeTest, cls).setUpClass()
    # Attempt to initialize the TPU system.
    try:
      cls.resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          tpu='local'
      )
    except Exception as e:
      raise tf.SkipTest("TPU not available; skipping multi-TPU test.") from e
    tf.config.experimental_connect_to_cluster(cls.resolver)
    tf.tpu.experimental.initialize_tpu_system(cls.resolver)
    cls.strategy = tf.distribute.TPUStrategy(cls.resolver)
    if cls.strategy.num_replicas_in_sync != 8:
      raise tf.SkipTest("Need 8 TPU replicas for this test.")

  def test_eight_core_exchange(self):
    # Test parameters.
    cx, cy, cz = 2, 2, 2
    n_cores = cx * cy * cz
    replicas = np.arange(n_cores, dtype=np.int32).reshape((cx, cy, cz))

    # This function will be executed on each replica.
    @tf.function
    def distributed_fn():

      replica_context = tf.distribute.get_replica_context()
      r = replica_context.replica_id_in_sync_group

      capacity = 10
      active_count = 8

      # Create per-replica integer field tensor.
      lpt_field_ints = tf.tile([[1, r]], [active_count, 1])
      lpt_field_ints = tf.pad(
          lpt_field_ints, [[0, capacity - active_count], [0, 0]]
      )

      # Create per-replica float field tensor.
      lpt_field_floats = tf.cast(
          r * tf.ones((active_count, 7), tf.int32), tf.float32
      )
      lpt_field_floats = tf.pad(
          lpt_field_floats, [[0, capacity - active_count], [0, 0]]
      )

      # Determining destination replicas. Each replica sends 1 particle to
      # every other replica.
      dest_replicas = tf.concat(
          (
              tf.range(active_count) % n_cores,
              tf.fill((capacity - active_count,), r),
          ),
          axis=0
      )

      # Call the neighbor_exchange function.
      return lpt_comm.neighbor_exchange(
          lpt_field_ints,
          lpt_field_floats,
          dest_replicas,
          r,
          replicas,
      )

    # Run the distributed function using TPUStrategy.
    tf.summary.trace_on(graph=True, profiler=True, profiler_outdir='./logs')
    distributed_results = self.strategy.run(distributed_fn)
    tf.summary.trace_export(
        name="neighbor_exchange_trace", step=0, profiler_outdir='./logs'
    )

    ints_results = self.strategy.experimental_local_results(
        distributed_results[0]
    )
    print(ints_results)
    floats_results = self.strategy.experimental_local_results(
        distributed_results[1]
    )

    neighbor_dirs = [
        (1, 0, 0),  # +x
        (-1, 0, 0),  # +x
        (0, 1, 0),  # +y
        (0, -1, 0),  # +y
        (0, 0, 1),  # +z
        (0, 0, -1),  # +z
    ]

    print("END CALCULATION", replicas)

    for i, j, k in itertools.product(range(cx), range(cy), range(cz)):
      r = replicas[i, j, k]
      ints_out = ints_results[r].numpy()
      floats_out = floats_results[r].numpy()

      # Get list of replicas this replica recieved data from.
      # Iterating through +x, -x, +y, -y, +z, -z.
      recv_from_lst = []
      for l, (p, q, w) in enumerate(neighbor_dirs):
        ci = (cx, cy, cz)[l // 2]
        if ci == 1 or (ci == 2 and l % 2 == 1):
          continue
        recv_from = replicas[(i - p) % cx, (j - q) % cy, (k - w) % cz]
        recv_from_lst.append(recv_from)

      # Check that the received data is correct.
      recv_from_lst = np.array([recv_from_lst] * 8)
      expected_ints = np.concatenate(
          (np.ones_like(recv_from_lst), recv_from_lst), axis=1
      )
      expected_floats = np.tile(recv_from_lst, (1, 7))

      ints_out = ints_out[ints_out[:, 1].argsort()]
      floats_out = floats_out[floats_out[:, 0].argsort()]
      expected_ints = expected_ints[expected_ints[:, 1].argsort()]
      expected_floats = expected_floats[expected_floats[:, 0].argsort()]

      with self.subTest(f"replica={r}"):
        self.assertAllEqual(expected_ints, ints_out)
        self.assertAllEqual(expected_floats, floats_out)


if __name__ == "__main__":
  tf1.logging.set_verbosity(tf1.logging.INFO)
  tf.profiler.experimental.server.start(6000)
  tf.test.main()
