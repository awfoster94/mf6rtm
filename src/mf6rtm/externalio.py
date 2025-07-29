import os
from mf6rtm.config import MF6RTMConfig

class Regenerator:
    """
    A class to regenerate a Mup3d object from a script file.
    """
    def __init__(self, wd='.', phinp='phinp.dat', nlay=1):
        self.wd = os.path.join(wd)
        self.phinp = phinp
        self.config = MF6RTMConfig.from_toml_file(os.path.join(self.wd, 'mf6rtm.toml')).to_dict()
        self.nlay = nlay


    def validate_external_files(self):
        """
        Validate the existence of external files required for regeneration.
        """
        phinp_path = os.path.join(self.wd, self.phinp)
        if not os.path.exists(phinp_path):
            raise FileNotFoundError(f"Required file '{self.phinp}' not found in working directory '{self.wd}'.")
        
        for key, value in self.config.items():
            if key is not 'reactive':
                print(f"Key: {key}")
                for lay in self.nlay:
                    file_path = os.path.join(self.wd, f"{key}.")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Required file '{value}' for key '{key}' not found in working directory '{self.wd}'.")

    def regenerate(self):
        # Implementation of regeneration logic goes here
        pass
