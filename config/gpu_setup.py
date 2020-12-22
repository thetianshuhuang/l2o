"""GPU setup."""

import tensorflow as tf
from tensorflow.config.experimental import VirtualDeviceConfiguration


def create_distribute(vgpus=1):
    """Create tf.distribute.strategy."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if vgpus > 1:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            VirtualDeviceConfiguration(memory_limit=512)
            for _ in range(vgpus)])
        print("Created {} Virtual GPUs:".format(vgpus))
        vgpu_list = tf.config.experimental.list_logical_devices('GPU')
        for i, d in enumerate(vgpu_list):
            print("  <{}> {}".format(i, str(d)))
    else:
        print("Using {} GPUs:".format(len(gpus)))
        for i, d in enumerate(gpus):
            print("  <{}> {}".format(i, str(d)))

    return tf.distribute.MirroredStrategy()
