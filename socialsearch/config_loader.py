import os
import yaml


class _Config:
    def __init__(self, yaml_path):
        with open (yaml_path, "r") as f:
            conf = yaml.safe_load(f)
        conf = self.__make_paths_absolute(os.path.dirname(yaml_path), conf)
        self.config = conf

    def __getattr__(self, name):
        if name in self.config:
            return self.config[name]
        else:
            raise KeyError(f"Config property '{name}' does not exist")

    def __make_paths_absolute(self, dir_, cfg):
        '''Make all values for keys ending with `_path` absolute to dir_'''
        for key in cfg.keys():
            if key.endswith("_path"):
                cfg[key] = os.path.join(dir_, cfg[key])
                cfg[key] = os.path.abspath(cfg[key])
                # if not os.path.isfile(cfg[key]):
                #     logging.error("%s does not exist.", cfg[key])
            if type(cfg[key]) is dict:
                cfg[key] = self.__make_paths_absolute(dir_, cfg[key])
        return cfg


config = _Config("config.yaml")