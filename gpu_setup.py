"""GPU setup."""

import tensorflow as tf
from tensorflow.config.experimental import VirtualDeviceConfiguration


def create_distribute(
        vgpus=1, memory_limit=512, gpu_idx=0, do_cpu=False, gpus=None):
    """Create tf.distribute.strategy."""
    if do_cpu:
        cpus = tf.config.experimental.list_physical_devices('CPU')
        return tf.distribute.MirroredStrategy(devices=["/CPU:0"])

    if gpus is not None:
        dev_list = tf.config.list_physical_devices('CPU')
        gpu_list = tf.config.list_physical_devices('GPU')
        for gpu in gpus.split(','):
            dev_list += gpu_list[int(gpu)]

    gpus = tf.config.experimental.list_visible_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if vgpus > 1:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[gpu_idx], [
                VirtualDeviceConfiguration(memory_limit=memory_limit)
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
