"""Adam Baseline."""

import json
import tensorflow as tf
import l2o


results = [
    l2o.evaluate.evaluate(tf.keras.optimizers.Adam()) for _ in range(10)]
results = {k: np.stack([d[k] for d in results]) for k in results[0]}
np.savez('baseline_adam.npz', **results)
