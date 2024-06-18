import os
from jupyter_server.services.config import ConfigManager

def get_config():
    cm = ConfigManager()
    cm.update('notebook', {
        "ContentsManager": {
            "extra_static_paths": [os.path.abspath("files")]
        }
    })

get_config()