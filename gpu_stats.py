import tensorflow as tf
import nvidia_smi

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

devices = session.list_devices()
for d in devices:
    print(d.name)

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)

print(tf.config.experimental.get_memory_usage("GPU:0"))  # 0

tensor_1_mb = tf.zeros((1, 256, 256, 1024), dtype=tf.float32)
# print(tf.config.experimental.get_memory_usage("GPU:0"))  # 1050112
gpu_mem = round(tf.config.experimental.get_memory_usage("GPU:0") / 1024 / 1024, 2)
print(r'gpu memory：total {} MB'.format(gpu_mem))


tensor_1_mb = None
print(tf.config.experimental.get_memory_usage("GPU:0"))  # 2098688

