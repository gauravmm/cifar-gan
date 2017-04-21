# Configuration for adversarial trainer 

import os

LOG_INTERVAL_DEFAULT = 500
NUM_BATCHES_DEFAULT = 5000

PATH = {
    "__main__": os.path.dirname(os.path.abspath(__file__)),
    "log"     : "experiment.log",
    "output"  : "train_logs",
    "cache"   : ".cache",
}

def _WEIGHT_FILENAME(typ="*", step="*") -> str:
    if step != "*":
        step = "{:06d}".format(step) # Pad with leading zeros
    return "checkpoint-{}-{}.h5".format(typ, step)

def _IMAGE_FILENAME(step="*") -> str:
    return "generated-{}.png".format(step)

# Dispatch the filename call as appropriate:
_FILENAME = {
    'weight': _WEIGHT_FILENAME,
     'image': _IMAGE_FILENAME, 
    'struct': lambda nm: "{}.png".format(nm),
       'csv': lambda: "loss.csv",
         ".": lambda: None
}
def get_filename(t, cli_args, *args):
    f = lambda x: ".".join(x.split(".")[1:])
    hstr = "{}-{}-{}-{}".format(f(cli_args.hyperparam.__name__),
                                f(cli_args.generator.__name__),
                                f(cli_args.discriminator.__name__),
                                "_".join(f(p.__name__) for p in cli_args.preprocessor))
    rv = _FILENAME[t](*args)
    if rv:
        return os.path.join(PATH["output"], hstr, rv)
    else:
        return os.path.join(PATH["output"], hstr)

IMAGE_GUTTER = 5
