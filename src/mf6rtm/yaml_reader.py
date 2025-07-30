import yaml
import numpy as np

# Add the path to phreeqcrm if needed
# sys.path.append('path/to/phreeqcrm')

try:
    from phreeqcrm import yamlphreeqcrm
except ImportError:
    print("Warning: phreeqcrm not found. This is a demonstration of the concept.")
    # Mock class for demonstration
    class YAMLPhreeqcRM:
        def __init__(self):
            self.yaml_doc = []
        def YAMLSetGridCellCount(self, count): pass
        def YAMLThreadCount(self, nthreads): pass
        def YAMLSetComponentH2O(self, tf): pass
        def YAMLUseSolutionDensityVolume(self, tf): pass
        def YAMLSetFilePrefix(self, prefix): pass
        def YAMLOpenFiles(self): pass
        def YAMLSetErrorHandlerMode(self, mode): pass
        def YAMLSetRebalanceFraction(self, f): pass
        def YAMLSetRebalanceByCell(self, tf): pass
        def YAMLSetPartitionUZSolids(self, tf): pass
        def YAMLSetUnitsSolution(self, option): pass
        def YAMLSetUnitsPPassemblage(self, option): pass
        def YAMLSetUnitsExchange(self, option): pass
        def YAMLSetUnitsSurface(self, option): pass
        def YAMLSetUnitsGasPhase(self, option): pass
        def YAMLSetUnitsSSassemblage(self, option): pass
        def YAMLSetUnitsKinetics(self, option): pass
        def YAMLSetPorosity(self, por): pass
        def YAMLSetPrintChemistryMask(self, cell_mask): pass
        def YAMLSetPrintChemistryOn(self, workers, initial_phreeqc, utility): pass
        def YAMLSetRepresentativeVolume(self, rv): pass
        def YAMLLoadDatabase(self, database): pass
        def YAMLRunFile(self, workers, initial_phreeqc, utility, chemistry_name): pass
        def YAMLRunString(self, workers, initial_phreeqc, utility, input_string): pass
        def YAMLAddOutputVars(self, option, definition): pass
        def YAMLFindComponents(self): pass
        def YAMLInitialPhreeqc2Module(self, ic): pass
        def YAMLRunCells(self): pass
        def YAMLSetTime(self, time): pass
        def WriteYAMLDoc(self, filename): pass
    
    yamlphreeqcrm = type('module', (), {'YAMLPhreeqcRM': YAMLPhreeqcRM})()

def load_yaml_to_phreeqcrm(yaml_file_path):
    """
    Load a YAML file and reconstruct a YAMLPhreeqcRM instance.
    
    Args:
        yaml_file_path (str): Path to the YAML file
        
    Returns:
        YAMLPhreeqcRM: Reconstructed instance
    """
    
    # Read the YAML file
    with open(yaml_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    
    # Create a new YAMLPhreeqcRM instance
    yrm = yamlphreeqcrm.YAMLPhreeqcRM()
    
    # Method mapping dictionary
    method_mapping = {
        'SetGridCellCount': lambda item: yrm.YAMLSetGridCellCount(item['count']),
        'ThreadCount': lambda item: yrm.YAMLThreadCount(item['nthreads']),
        'SetComponentH2O': lambda item: yrm.YAMLSetComponentH2O(item['tf']),
        'UseSolutionDensityVolume': lambda item: yrm.YAMLUseSolutionDensityVolume(item['tf']),
        'SetFilePrefix': lambda item: yrm.YAMLSetFilePrefix(item['prefix']),
        'OpenFiles': lambda item: yrm.YAMLOpenFiles(),
        'SetErrorHandlerMode': lambda item: yrm.YAMLSetErrorHandlerMode(item['mode']),
        'SetRebalanceFraction': lambda item: yrm.YAMLSetRebalanceFraction(item['f']),
        'SetRebalanceByCell': lambda item: yrm.YAMLSetRebalanceByCell(item['tf']),
        'SetPartitionUZSolids': lambda item: yrm.YAMLSetPartitionUZSolids(item['tf']),
        'SetUnitsSolution': lambda item: yrm.YAMLSetUnitsSolution(item['option']),
        'SetUnitsPPassemblage': lambda item: yrm.YAMLSetUnitsPPassemblage(item['option']),
        'SetUnitsExchange': lambda item: yrm.YAMLSetUnitsExchange(item['option']),
        'SetUnitsSurface': lambda item: yrm.YAMLSetUnitsSurface(item['option']),
        'SetUnitsGasPhase': lambda item: yrm.YAMLSetUnitsGasPhase(item['option']),
        'SetUnitsSSassemblage': lambda item: yrm.YAMLSetUnitsSSassemblage(item['option']),
        'SetUnitsKinetics': lambda item: yrm.YAMLSetUnitsKinetics(item['option']),
        'SetPorosity': lambda item: yrm.YAMLSetPorosity(item['por']),
        'SetPrintChemistryMask': lambda item: yrm.YAMLSetPrintChemistryMask(item['cell_mask']),
        'SetPrintChemistryOn': lambda item: yrm.YAMLSetPrintChemistryOn(
            item['workers'], item['initial_phreeqc'], item['utility']
        ),
        'SetRepresentativeVolume': lambda item: yrm.YAMLSetRepresentativeVolume(item['rv']),
        'LoadDatabase': lambda item: yrm.YAMLLoadDatabase(item['database']),
        'RunFile': lambda item: yrm.YAMLRunFile(
            item['workers'], item['initial_phreeqc'], item['utility'], item['chemistry_name']
        ),
        'RunString': lambda item: yrm.YAMLRunString(
            item['workers'], item['initial_phreeqc'], item['utility'], item['input_string']
        ),
        'AddOutputVars': lambda item: yrm.YAMLAddOutputVars(item['option'], item['definition']),
        'FindComponents': lambda item: yrm.YAMLFindComponents(),
        'InitialPhreeqc2Module': lambda item: yrm.YAMLInitialPhreeqc2Module(item['ic']),
        'RunCells': lambda item: yrm.YAMLRunCells(),
        'SetTime': lambda item: yrm.YAMLSetTime(item['time']),
    }
    not_to_read = ['RunFile', 'RunString', 'AddOutputVars', 'FindComponents', 'RunCells', 'SetTime']
    # Process each item in the YAML data
    for item in yaml_data:
        key = item.get('key')
        if key in method_mapping:
            if key == 'InitialPhreeqc2Module':
                # Handle the initial conditions separately
                ic1 = np.array(item.get('ic', []))
                continue
            elif key in not_to_read:
                continue
            try:
                method_mapping[key](item)
                # print(f"✓ Processed: {key}")
            except Exception as e:
                print(f"Error processing {key}: {e}")


        else:
            print(f"Unknown key: {key}")
    
    return yrm, ic1

def simple_yaml_reader(yaml_file_path):
    """
    Simple example of reading YAML data without reconstruction.
    
    Args:
        yaml_file_path (str): Path to the YAML file
        
    Returns:
        dict: Parsed YAML data
    """
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    print("YAML File Contents:")
    print("=" * 50)
    
    for i, item in enumerate(data):
        print(f"Item {i+1}:")
        print(f"  Key: {item.get('key', 'N/A')}")
        
        # Print other parameters
        for k, v in item.items():
            if k != 'key':
                if isinstance(v, list) and len(v) > 5:
                    print(f"  {k}: [{v[0]}, {v[1]}, ..., {v[-1]}] (length: {len(v)})")
                else:
                    print(f"  {k}: {v}")
        print()
    
    return data

def extract_specific_data(yaml_file_path, key_name):
    """
    Extract specific data from YAML file by key name.
    
    Args:
        yaml_file_path (str): Path to the YAML file
        key_name (str): Key to search for
        
    Returns:
        dict or None: Found item or None
    """
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    for item in data:
        if item.get('key') == key_name:
            return item
    
    return None

# Example usage
if __name__ == "__main__":
    yaml_file = "ex5/mf6rtm/mf6rtm.yaml"
    
    print("Example 1: Simple YAML Reading")
    print("=" * 60)
    try:
        data = simple_yaml_reader(yaml_file)
        print(f"Successfully read {len(data)} items from YAML file")
    except FileNotFoundError:
        print(f"File {yaml_file} not found")
    except Exception as e:
        print(f"Error reading YAML: {e}")
    
    print("\n" + "=" * 60)
    print("Example 2: Extract Specific Data")
    print("=" * 60)
    
    # Extract porosity data
    porosity_data = extract_specific_data(yaml_file, "SetPorosity")
    if porosity_data:
        print("Porosity data found:")
        print(f"  Values: {porosity_data['por'][:5]}... (showing first 5)")
        print(f"  Total cells: {len(porosity_data['por'])}")
    
    # Extract database info
    db_data = extract_specific_data(yaml_file, "LoadDatabase")
    if db_data:
        print(f"\nDatabase: {db_data['database']}")
    
    print("\n" + "=" * 60)
    print("Example 3: Reconstruct YAMLPhreeqcRM Instance")
    print("=" * 60)
    
    try:
        reconstructed_yrm = load_yaml_to_phreeqcrm(yaml_file)
        print("Successfully reconstructed YAMLPhreeqcRM instance!")
        
        # You can now use the reconstructed instance
        # For example, write it to a new file:
        # reconstructed_yrm.WriteYAMLDoc("reconstructed_config.yaml")
        
    except Exception as e:
        print(f"Error reconstructing YAMLPhreeqcRM: {e}")

    print("\n" + "=" * 60)
    print("Example 4: Working with Specific Arrays")
    print("=" * 60)
    
    # Example of working with array data
    try:
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        
        # Find and work with porosity data
        for item in data:
            if item.get('key') == 'SetPorosity':
                porosity_values = item['por']
                print(f"Porosity statistics:")
                print(f"  Min: {min(porosity_values)}")
                print(f"  Max: {max(porosity_values)}")
                print(f"  Average: {sum(porosity_values)/len(porosity_values):.3f}")
                break
        
        # Find and work with initial conditions
        for item in data:
            if item.get('key') == 'InitialPhreeqc2Module':
                ic_values = item['ic']
                unique_values = set(ic_values)
                print(f"\nInitial conditions statistics:")
                print(f"  Total cells: {len(ic_values)}")
                print(f"  Unique values: {sorted(unique_values)}")
                break
                
    except Exception as e:
        print(f"Error processing arrays: {e}")
