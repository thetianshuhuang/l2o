import json
import tensorflow as tf
import l2o


l2o.evaluate.evaluate(tf.keras.optimizers.Adam())
with open("baseline.json") as f:
    json.dump(results, f)
