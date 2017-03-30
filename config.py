# Configuration for adversarial trainer 

import os

LOG_INTERVAL_DEFAULT = 500

PATH = {
    "__main__": os.path.dirname(os.path.abspath(__file__)),
    "weights" : "weights",
    "logs"    : "train_logs",
    "images"  : "train_logs",
    "cache"   : ".cache",
    "data"    : "data",
}

def _WEIGHT_FILENAME(g_name : str, d_name : str, typ : str = "*", step=None) -> str:
    if step is not None:
        step = "{:06d}".format(step) # Pad with leading zeros
    else:
        step = "*"                   # Wildcard
    
    return os.path.join(PATH["weights"], "checkpoint-{}-{}-{}-{}.h5".format(g_name, d_name, typ, step))

def _IMAGE_FILENAME(g_name : str, d_name : str, step="*") -> str:
    return os.path.join(PATH["images"], "generated-{}-{}-{}.png".format(g_name, d_name, step))

def _CSV_FILENAME(g_name : str, d_name : str) -> str:
    return os.path.join(PATH["logs"], "loss-{}-{}.csv".format(g_name, d_name))

# Dispatch the filename call as appropriate:
_FILENAME = {'weight': _WEIGHT_FILENAME, 'image': _IMAGE_FILENAME, 'csv': _CSV_FILENAME}
def get_filename(t, cli_args, *args):
    return _FILENAME[t](cli_args.generator.NAME, cli_args.discriminator.NAME, *args)

IMAGE_GUTTER = 5