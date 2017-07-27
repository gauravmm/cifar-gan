# Configuration for adversarial trainer 

import os

LOG_INTERVAL_DEFAULT = 500
NUM_BATCHES_DEFAULT = 5000

PATH = {
    "__main__": os.path.dirname(os.path.abspath(__file__)),
    "log"     : "experiment.log",
    "output"  : "train_logs_1_1_1_dropout",
    "cache"   : ".cache",
}

# Dispatch the filename call as appropriate:
_FILENAME = {
    'struct': lambda nm: "{}.png".format(nm),
         ".": lambda: None,
"model.ckpt": lambda: "model.ckpt"
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
