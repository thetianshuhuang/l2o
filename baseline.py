import json
import tensorflow as tf
import l2o


results = l2o.evaluate.evaluate(tf.keras.optimizers.Adam())
with open("baseline.json", "w") as f:
    json.dump(results, f)
