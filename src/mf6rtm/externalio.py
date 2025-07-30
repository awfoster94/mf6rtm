import os
import numpy as np
from mf6rtm.config import MF6RTMConfig
from mf6rtm.mf6api import Mf6API
from mf6rtm.discretization import grid_dimensions, total_cells_in_grid
from mf6rtm.yaml_reader import load_yaml_to_phreeqcrm

ic_position = {
    'equilibrium_phases': 1, 
    'exchange_phases': 2, 
    'surface_phases': 3,
    'gas_phases': 4,
    'solid_solution_phases': 5,
     'kinetic_phases':6,
}

class Regenerator:
    """
    A class to regenerate a Mup3d object from a script file.
    """
    def __init__(self, wd='.', phinp='phinp.dat'):
        self.wd = os.path.abspath(wd)
        self.yamlfile = os.path.join(self.wd, 'mf6rtm.yaml')
        self.phinp = phinp
        self.config = MF6RTMConfig.from_toml_file(os.path.join(self.wd, 'mf6rtm.toml')).to_dict()
        self.grid_shape = grid_dimensions(Mf6API(self.wd, os.path.join(self.wd, 'libmf6.dll')))
        self.nlay = self.grid_shape[0]
        self.nxyz = total_cells_in_grid(Mf6API(self.wd, os.path.join(self.wd, 'libmf6.dll')))

        # self.validate_external_files()

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
                names = self.config[key]['names'] if 'names' in self.config[key] else ValueError(f"Key '{key}' does not have 'names' attribute.") 
                for nme in names:
                    print(f"Name: {nme}")
                    for lay in range(self.nlay):
                        print(f"Layer: {lay}")
                        file_path = os.path.join(self.wd, f"{key}.{nme}.m0.layer{lay+1}.txt")
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(f"Required file '{file_path}' for key '{key}' not found in working directory '{self.wd}'.")

    def read_phinp(self):
        with open(os.path.join(self.wd, self.phinp), 'r') as f:
            script = f.readlines()
        return script
    
    def get_solution_blocks(self, script):
        """
        Extract solution blocks from the script.
        """
        block = []
        in_postfix = False
        for line in script:
            if line.startswith('EQUILIBRIUM') or line.startswith('KINETIC') or line.startswith('EXCHANGE'):
                in_postfix = False
            if line.startswith('SOLUTION'):
                in_postfix = True
            if in_postfix:
                block.append(line)

        # set new attibute named self.solution_blocks
        self.solution_blocks = block
        return block

    def update_yaml(self, filename='mf6rtm.yaml'):

        """Update the YAML file with the regenerated script and initial conditions.
        """
        yamlphreeqcrm, ic1 = load_yaml_to_phreeqcrm(self.yamlfile)
        ic1 = ic1.reshape(7, self.nxyz).T
        ic1_phases = np.reshape(np.arange(1, self.nxyz + 1), self.nxyz)

        phases = [i for i in self.config.keys() if 'phases' in i]

        for phase in phases:
            i = ic_position[phase]
            ic1[:, i] = ic1_phases

        ic1_flatten = ic1.flatten('F')

        status = yamlphreeqcrm.YAMLRunFile(True, True, True, self.regenerated_phinp)
        # Clear contents of workers and utility
        input = "DELETE; -all"
        status = yamlphreeqcrm.YAMLRunString(True, False, True, input)
        yamlphreeqcrm.YAMLAddOutputVars("AddOutputVars", "true")

        status = yamlphreeqcrm.YAMLFindComponents()
        status = yamlphreeqcrm.YAMLInitialPhreeqc2Module(ic1_flatten)
        status = yamlphreeqcrm.YAMLRunCells()
        # Initial equilibration of cells
        time = 0.0
        status = yamlphreeqcrm.YAMLSetTime(time)

        # status = yamlphreeqcrm.YAMLRunFile(True, True, True, self.regenerated_phinp)
        # status = yamlphreeqcrm.YAMLInitialPhreeqc2Module(ic1_flatten)
        fdir = os.path.join(self.wd, filename)
        status = yamlphreeqcrm.WriteYAMLDoc(fdir)

        return ic1_flatten

    def get_postfix_block(self, script):
        """
        Extract the postfix block from the script.
        """
        postfix_block = []
        in_postfix = False
        for line in script:
            if line.startswith('SELECTED_OUTPUT'):
                in_postfix = True
            if in_postfix:
                postfix_block.append(line)
            # if line.strip() == 'END':
                # in_postfix = False
        postfix_block = ['PRINT\n'] + postfix_block  # Ensure it starts with 'PRINT\n'
        self.postfix_blocks = postfix_block
        # + ''.join(postfix_block).strip()
        return postfix_block
    
    def generate_new_script(self):
        """
        Generate a new script based on the existing script and configuration.
        """
        script = self.read_phinp()
        solution_blocks = self.get_solution_blocks(script)
        postfix_block = self.get_postfix_block(script)

        # Create a new script with the solution blocks and postfix block
        new_script = []
        new_script.extend(solution_blocks)

        # Add equilibrium phases, kinetic phases, and exchange blocks
        new_script.extend(self.generate_equilibrium_phases_blocks())
        new_script.extend(self.generate_kinetic_phases_blocks())
        new_script.extend(self.generate_exchange_blocks())
        new_script.extend(postfix_block)
        self.regenerated_script = ''.join(new_script).strip()
        return self.regenerated_script

    def write_new_script(self, filename='reg_phinp.dat'):
        """
        Write the regenerated script to a file.
        """
        if not hasattr(self, 'regenerated_script'):
            self.generate_new_script()
        with open(os.path.join(self.wd, filename), 'w') as f:
            f.write(self.regenerated_script)
        print(f"New script written to {os.path.join(self.wd, filename)}")
        self.regenerated_phinp = os.path.join(self.wd, filename)
        return self.regenerated_phinp
    


    def generate_equilibrium_phases_blocks(self):
        """
        Generate equilibrium phases blocks from the config.
        """
        self.add_m0_to_config()
        equilibrium_phases = self.config.get('equilibrium_phases', {})
        blocks = []

        n_phases = self.nxyz
        for i_phase in range(1, n_phases+1):
            block = f"EQUILIBRIUM_PHASES {i_phase}\n"
            for nme in equilibrium_phases['names']:
                si = equilibrium_phases.get(f'si', None).get(nme, None)
                m0 = equilibrium_phases.get(f'm0', None).get(nme, None).flatten()
                block += f"    {nme} {si:.5e} {m0[i_phase-1]:.5e}\n"
            block += "END\n"
            blocks.append(block)
        self.equilibrium_phases_blocks = blocks
        return blocks

    def generate_kinetic_phases_blocks(self):
        """
        Generate kinetic phases blocks from the config.
        """
        self.add_m0_to_config()
        kinetic_phases = self.config.get('kinetic_phases', {})
        blocks = []

        n_phases = self.nxyz
        for i_phase in range(1, n_phases+1):
            block = f"KINETICS {i_phase}\n"
            for nme in kinetic_phases['names']:
                # Get parameters for this kinetic phase
                parms = kinetic_phases.get('parms', {}).get(nme, [])
                m0 = kinetic_phases.get('m0', {}).get(nme, None).flatten()
                
                # Start the kinetic phase line with name and initial moles
                block += f"    {nme}\n"
                block += f"        -m0 {m0[i_phase-1]:.5e}\n"
                
                # Add parameters if they exist
                if parms:
                    parms_str = " ".join([f"{p:.5e}" for p in parms])
                    block += f"        -parms {parms_str}\n"
                
                # Add formula if it exists
                formula = kinetic_phases.get('formula', {}).get(nme, None)
                if formula:
                    block += f"        -formula {formula}\n"
            block += "END\n"
            blocks.append(block)
        self.kinetic_phases_blocks = blocks
        return blocks

    def generate_exchange_blocks(self):
        """
        Generate exchange blocks from the config.
        """
        self.add_m0_to_config()
        exchange = self.config.get('exchange_phases', {})
        blocks = []

        n_phases = self.nxyz
        for i_phase in range(1, n_phases+1):
            block = f"EXCHANGE {i_phase}\n"
            for nme in exchange['names']:
                m0 = exchange.get('m0', {}).get(nme, None).flatten()
                block += f"    {nme} {m0[i_phase-1]:.5e}\n"
            # Hard code equilibrate 1 as requested
            block += "    -equilibrate 1\n"
            block += "END\n"
            blocks.append(block)
        self.exchange_blocks = blocks
        return blocks

    def read_external_files(self):
        """
        Read the external files required for regeneration using numpy.
        Returns a dictionary with the loaded arrays organized by key, name, and layer.
        """
        file_data = {}
        
        # Read phase files following the same logic as validate_external_files
        for key, value in self.config.items():
            if key != 'reactive':
                # print(f"Processing key: {key}")
                
                if 'names' not in self.config[key]:
                    print(f"Warning: Key '{key}' does not have 'names' attribute, skipping.")
                    continue
                    
                names = self.config[key]['names']
                file_data[key] = {}
                
                for nme in names:
                    # print(f"Processing name: {nme}")
                    layer_arrays = []
                    
                    # Load all layers for this name
                    for lay in range(self.nlay):
                        # print(f"Processing layer: {lay + 1}")
                        file_path = os.path.join(self.wd, f"{key}.{nme}.m0.layer{lay+1}.txt")
                        
                        if os.path.exists(file_path):
                            try:
                                # Load the array using numpy
                                array_data = np.loadtxt(file_path)
                                layer_arrays.append(array_data)
                                # print(f"Loaded array from {file_path}, shape: {array_data.shape}")
                            except Exception as e:
                                print(f"Warning: Could not load file {file_path}: {e}")
                                layer_arrays.append(None)
                        else:
                            print(f"Warning: File {file_path} does not exist")
                            layer_arrays.append(None)
                    
                    # Merge layers and reshape using grid dimensions
                    if any(arr is not None for arr in layer_arrays):
                        try:
                            # Filter out None values and stack the arrays
                            valid_arrays = [arr for arr in layer_arrays if arr is not None]
                            if valid_arrays:
                                # Stack arrays along the first axis (layers)
                                merged_array = np.stack(valid_arrays, axis=0)
                                
                                # Reshape to grid dimensions: (nlay, nrow, ncol)
                                nlay, nrow, ncol = self.grid_shape
                                reshaped_array = merged_array.reshape(nlay, nrow, ncol)
                                
                                file_data[key][nme] = reshaped_array
                                # print(f"Merged and reshaped {nme} to shape: {reshaped_array.shape}")
                            else:
                                file_data[key][nme] = None
                                print(f"Warning: No valid arrays found for {nme}")
                        except Exception as e:
                            print(f"Warning: Could not merge/reshape arrays for {nme}: {e}")
                            file_data[key][nme] = None
                    else:
                        file_data[key][nme] = None
                        print(f"Warning: No arrays loaded for {nme}")
        
        # Store the loaded data as an instance attribute
        self.file_data = file_data
        return file_data

    def add_m0_to_config(self):
        """
        Add the loaded array data to the config dictionary.
        This method should be called after read_external_files().
        """
        if not hasattr(self, 'file_data'):
            self.read_external_files()
        
        # Add phase array data to config
        for key in self.file_data:
            if key != 'phinp' and key in self.config:
                # Add arrays section to each phase type
                if 'm0' not in self.config[key]:
                    self.config[key]['m0'] = {}
                
                self.config[key]['m0'] = self.file_data[key]
                # print(f"Added m0 data for {key} to config")
        
        return self.config

    def regenerate(self):
        # Implementation of regeneration logic goes here
        pass
