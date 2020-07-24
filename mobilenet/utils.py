import os
import os.path as path


def ensure_exists(dirname):
    if not path.isdir(dirname):
        os.makedirs(dirname)
