#----------------------------------------------------------------------------
# © 2024 – UMONS
#
# Created By  : Sukanya Patra
# Created Date: 11-Mar-2024
# version ='1.0'
# ---------------------------------------------------------------------------

import json

class Config(object):
    """Base class for experimental setting/configuration."""

    def __init__(self, settings):
        self.settings = settings

    def load_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():

            if isinstance(value, dict):
                for param in value.keys():
                    if isinstance(value[param], str):
                        value[param] = value[param].lower()

            elif isinstance(value, str):
                value = value.lower()

            self.settings[key] = value

    def save_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w') as fp:
            json.dump(self.settings, fp)

    def formatted_config(self):

        formatted_json = dict()
        for key, value in self.settings.items():
            formatted_json[key] = str(value)

        return formatted_json
