import dataclasses
from waymax import dataloader
import waymax.config as cfg

# Base config preset (works with your installed waymax)
base = cfg.WOD_1_0_0_TESTING

# You have 10 local TFRecord shards in this folder
LOCAL_PATTERN = "/home/med1e/vmax/data/waymo_raw/training/*.tfrecord@10"

# Build a local config (important: use dataclasses.replace because .replace() doesn't exist)
conf = dataclasses.replace(
    base,
    path=LOCAL_PATTERN,
    batch_dims=(1,),     # batch size 1
    num_shards=1,        # keep it simple locally
    repeat=1,            # one pass for the test
    shuffle_seed=None,   # deterministic
    deterministic=True,
)

gen = dataloader.simulator_state_generator(conf)
batch = next(gen)

print("SUCCESS ✅")
print("Type:", type(batch))
print("Some attrs:", dir(batch)[:30])
