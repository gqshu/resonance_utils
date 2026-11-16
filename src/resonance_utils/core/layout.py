"""On-disk layout shared by every soundscape type.

    build/<type>/<id>/...            what `ss gen <type>` writes (one folder per soundscape)
    deploy/<type>/manifest.json      what `ss deploy <type>` writes, one manifest per type
    deploy/index.json                the packs present in deploy/ (written by every deploy)
"""

import os


def type_dir(root, type_name):
    """<root>/<type>; root is the build or deploy root the user passed."""
    return os.path.join(root, type_name)
