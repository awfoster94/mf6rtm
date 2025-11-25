import pytest
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from mf6rtm.mup3d.base import (
    Block,
    Solutions,
    EquilibriumPhases,
    ExchangePhases,
    KineticPhases,
    Surfaces,
    GasPhase,
    ChemStress,
    Mup3d,
    working_dir
)


# ==================== Fixtures ====================

@pytest.fixture
def sample_solutions_data():
    """Sample solutions data dictionary."""
    return {
        # 'Ca': [1.0, 2.0, 3.0],
        'Cl': [2.0, 4.0, 6.0],
        'pH': [7.0, 7.5, 8.0]
    }


@pytest.fixture
def sample_equilibrium_data():
    """Sample equilibrium phases data."""
    return {
        0: {'Calcite': {'m0': 0.0, 'si': 0.5}},
        1: {'Gypsum': {'m0': 1.0, 'si': -0.5}}
    }


@pytest.fixture
def sample_kinetic_data():
    """Sample kinetic phases data."""
    return {
        0: {
            'Pyrite': {
                'm0': 0.1,
                'parms': [1e-5, 2.0],
                'formula': 'FeS2',
                'steps': [100, 200]
            }
        }
    }


@pytest.fixture
def sample_exchange_data():
    """Sample exchange phases data."""
    return {
        0: {'X': {'m0': 1.5e-3}},
        1: {'X': {'m0': 2.0e-3}}
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)

@pytest.fixture
def db_temp_dir():
    """Create a temporary directory for database testing."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)

@pytest.fixture
def mock_database(db_temp_dir):
    """Create a mock database file."""
    db_path = os.path.join(db_temp_dir, 'pht3d_datab.dat')
    db_content = """SOLUTION_MASTER_SPECIES
H		H+	-1.0	H		1.008
H(0)		H2	0	H
H(1)		H+	-1.0	0
Ca		Ca+2	0	Ca		40.08
Cl		Cl-	0	Cl		35.453

SOLUTION_SPECIES
H+ = H+
	-gamma	9.0	0
	-dw	9.31e-9  1000  0.46  1e-10 # The dw parameters are defined in ref. 3.
# Dw(TK) = 9.31e-9 * exp(1000 / TK - 1000 / 298.15) * TK * 0.89 / (298.15 * viscos)
# Dw(I) = Dw(TK) * exp(-0.46 * DH_A * |z_H+| * I^0.5 / (1 + DH_B * I^0.5 * 1e-10 / (1 + I^0.75)))
e- = e-
H2O = H2O
# H2O + 0.01e- = H2O-0.01; -log_k -9 # aids convergence
Ca+2 = Ca+2
	-gamma	5.0	0.1650
	-dw	0.793e-9  97  3.4  24.6
	-Vm  -0.3456  -7.252  6.149  -2.479  1.239  5  1.60  -57.1  -6.12e-3  1 # ref. 1
Cl- = Cl-
	-gamma	3.5	  0.015
	-gamma	3.63  0.017 # cf. pitzer.dat
	-dw	2.03e-9  194  1.6  6.9
	-Vm  4.465  4.801  4.325  -2.847  1.748  0  -0.331  20.16  0  1 # ref. 1

PHASES
Calcite    CaCO3 = Ca+2 + CO3-2
Gypsum     CaSO4:2H2O = Ca+2 + SO4-2 + 2H2O
Pyrite     FeS2 + 3.5O2 + H2O = Fe+2 + 2SO4-2 + 2H+
END
EXCHANGE_MASTER_SPECIES
X    X-
END
SURFACE_MASTER_SPECIES
Hfo_w    Hfo_wOH
END"""
    with open(db_path, 'w') as f:
        f.write(db_content)
    return db_path


# ==================== Specialized Block Tests ====================

class TestSolutions:
    """Test suite for Solutions class."""
    
    def test_solutions_initialization(self, sample_solutions_data):
        """Test Solutions initialization."""
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(1)
        assert solutions.data == sample_solutions_data
        assert 'pH' in solutions.names
        assert 'Cl' in solutions.names


class TestEquilibriumPhases:
    """Test suite for EquilibriumPhases class."""
    
    def test_equilibrium_phases_initialization(self, sample_equilibrium_data):
        """Test EquilibriumPhases initialization."""
        eq_phases = EquilibriumPhases(sample_equilibrium_data)
        assert 'Calcite' in eq_phases.names
        assert 'Gypsum' in eq_phases.names
    
    @patch('mf6rtm.mup3d.base.utils.fill_missing_minerals')
    def test_fill_missing_minerals_called(self, mock_fill, sample_equilibrium_data):
        """Test that fill_missing_minerals is called."""
        eq_phases = EquilibriumPhases(sample_equilibrium_data)
        mock_fill.assert_called_once()


class TestKineticPhases:
    """Test suite for KineticPhases class."""
    
    def test_kinetic_phases_initialization(self, sample_kinetic_data):
        """Test KineticPhases initialization."""
        kinetic = KineticPhases(sample_kinetic_data)
        assert 'Pyrite' in kinetic.names
        assert kinetic.parameters is None
    
    def test_set_parameters(self, sample_kinetic_data):
        """Test setting kinetic parameters."""
        kinetic = KineticPhases(sample_kinetic_data)
        params = {'rate': 1e-5}
        kinetic.set_parameters(params)
        assert kinetic.parameters == params


class TestExchangePhases:
    """Test suite for ExchangePhases class."""
    
    def test_exchange_phases_initialization(self, sample_exchange_data):
        """Test ExchangePhases initialization."""
        exchange = ExchangePhases(sample_exchange_data)
        assert 'X' in exchange.names


class TestSurfaces:
    """Test suite for Surfaces class."""
    
    def test_surfaces_initialization(self):
        """Test Surfaces initialization."""
        surface_data = {0: {'Hfo': [0.1, 600]}}
        surfaces = Surfaces(surface_data)
        assert 'Hfo' in surfaces.names


# ==================== ChemStress Tests ====================

class TestChemStress:
    """Test suite for ChemStress class."""
    
    def test_chem_stress_initialization(self):
        """Test ChemStress initialization."""
        stress = ChemStress('WEL-1')
        assert stress.packnme == 'WEL-1'
        assert stress.sol_spd is None
        assert stress.packtype is None
    
    def test_set_spd(self):
        """Test setting stress period data."""
        stress = ChemStress('WEL-1')
        spd = [1, 2, 3, 4, 5]
        stress.set_spd(spd)
        assert stress.sol_spd == spd
    
    def test_set_packtype(self):
        """Test setting package type."""
        stress = ChemStress('WEL-1')
        stress.set_packtype('WEL')
        assert stress.packtype == 'WEL'


# ==================== Mup3d Tests ====================

class TestMup3dInitialization:
    """Test suite for Mup3d initialization."""
    
    def test_mup3d_initialization_dis(self, sample_solutions_data):
        """Test Mup3d initialization with DIS grid."""
        sol_ic = np.array([1, 2, 2, 3])
        sol_ic = np.reshape(sol_ic, (1, 2, 2))
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(sol_ic)
            
        model = Mup3d(
            solutions=solutions,
            nlay=1,
            nrow=2,
            ncol=2
        )
        assert model.nlay == 1
        assert model.nrow == 2
        assert model.ncol == 2
        assert model.nxyz == 4
        assert model.grid_shape == (1, 2, 2)
    
    def test_mup3d_initialization_disv(self, sample_solutions_data):
        """Test Mup3d initialization with DISV grid."""
        sol_ic = np.array([1, 2, 2, 3])
        sol_ic = np.reshape(sol_ic, (1, 4))
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(sol_ic)
        model = Mup3d(
            solutions=solutions,
            nlay=1,
            ncpl=4
        )
        assert model.nlay == 1
        assert model.ncpl == 4
        assert model.nxyz == 4
        assert model.grid_shape == (1, 4)
    
    # def test_mup3d_initialization_disu(self, sample_solutions_data):
    #     """Test Mup3d initialization with DISU grid."""
    #     solutions = Solutions(sample_solutions_data)
    #     model = Mup3d(
    #         solutions=solutions,
    #         nxyz=100
    #     )
    #     assert model.nxyz == 100
    #     assert model.grid_shape == (100,)
    
    def test_mup3d_no_solutions_error(self):
        """Test error when solutions not provided."""
        with pytest.raises(ValueError, match="solutions parameter is required"):
            Mup3d(nlay=1, nrow=1, ncol=1)
    
    def test_mup3d_no_grid_params_error(self, sample_solutions_data):
        """Test error when grid parameters not provided."""
        solutions = Solutions(sample_solutions_data)
        with pytest.raises(ValueError):
            Mup3d(solutions=solutions)
    
    def test_mup3d_old_style_initialization(self, sample_solutions_data):
        """Test old-style initialization (solutions as first arg)."""
        sol_ic = np.array([1, 2, 2, 3])
        sol_ic = np.reshape(sol_ic, (1, 1, 4))
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions, nlay=1, nrow=1, ncol=4)
        assert model.solutions == solutions
        assert model.nxyz == 4

class TestMup3dSetters:
    """Test suite for Mup3d setter methods."""
    
    def test_set_wd(self, sample_solutions_data, temp_dir):
        """Test setting working directory."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        model.set_wd(temp_dir)
        assert model.wd == Path(temp_dir)
        assert os.path.exists(temp_dir)
    
    def test_set_database(self, sample_solutions_data, temp_dir, mock_database):
        """Test setting database."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=10)
        model.set_wd(temp_dir)
        model.set_database(mock_database)
        # Database should be copied to wd
        assert os.path.exists(os.path.join(temp_dir, os.path.basename(mock_database)))
    
    def test_set_initial_temp(self, sample_solutions_data):
        """Test setting initial temperature."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        model.set_initial_temp(30.0)
        assert model.init_temp == 30.0
    
    def test_set_initial_temp_invalid(self, sample_solutions_data):
        """Test setting initial temperature with invalid type."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        with pytest.raises(AssertionError):
            model.set_initial_temp("invalid")
    
    def test_set_componenth2o(self, sample_solutions_data):
        """Test setting componentH2O flag."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=10)
        result = model.set_componenth2o(True)
        assert result is True
        assert model.componenth2o is True
    
    def test_set_componenth2o_invalid(self, sample_solutions_data):
        """Test set_componenth2o with invalid type."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        with pytest.raises(AssertionError):
            model.set_componenth2o("invalid")
    
    def test_get_componenth2o(self, sample_solutions_data):
        """Test getting componentH2O flag."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        model.set_componenth2o(True)
        assert model.get_componenth2o() is True
    
    def test_set_charge_offset(self, sample_solutions_data):
        """Test setting charge offset."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        model.set_charge_offset(10.0)
        assert model.charge_offset == 10.0
    
    def test_set_fixed_components(self, sample_solutions_data):
        """Test setting fixed components."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        fixed = ['H', 'O']
        model.set_fixed_components(fixed)
        assert model.fixed_components == fixed


class TestMup3dPhases:
    """Test suite for Mup3d phase-related methods."""
    
    def test_set_equilibrium_phases(self, sample_solutions_data, sample_equilibrium_data):
        """Test setting equilibrium phases."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=2, ncol=5)
        
        eq_phases = EquilibriumPhases(sample_equilibrium_data)
        eq_phases.set_ic(np.ones((1, 2, 5), dtype=int))
        model.set_equilibrium_phases(eq_phases)
        
        assert model.equilibrium_phases is not None
        assert model.equilibrium_phases.ic.shape == (1, 2, 5)
    
    def test_set_phases_kinetic(self, sample_solutions_data, sample_kinetic_data):
        """Test setting kinetic phases."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=2, ncol=5)
        
        kinetic = KineticPhases(sample_kinetic_data)
        kinetic.set_ic(np.zeros((1, 2, 5), dtype=int))
        model.set_phases(kinetic)
        
        assert model.kinetic_phases is not None
    
    def test_set_phases_invalid_ic_shape(self, sample_solutions_data, sample_equilibrium_data):
        """Test set_phases with invalid IC shape."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=2, ncol=5)
        
        eq_phases = EquilibriumPhases(sample_equilibrium_data)
        eq_phases.set_ic(np.ones((2, 3, 4), dtype=int))  # Wrong shape
        
        with pytest.raises(AssertionError):
            model.set_equilibrium_phases(eq_phases)
    
    def test_set_exchange_phases(self, sample_solutions_data, sample_exchange_data):
        """Test setting exchange phases."""
        solutions = Solutions(sample_solutions_data)
        sol_ic = 1
        solutions.set_ic(sol_ic)
        model = Mup3d(solutions=solutions, nlay=1, nrow=2, ncol=5)
        
        exchange = ExchangePhases(sample_exchange_data)
        exchange.set_ic(np.ones((1, 2, 5), dtype=int))
        model.set_exchange_phases(exchange)
        
        assert model.exchange_phases is not None


# class TestMup3dChemStress:
#     """Test suite for ChemStress in Mup3d."""
    
#     def test_set_chem_stress(self, sample_solutions_data):
#         """Test setting ChemStress."""
#         solutions = Solutions(sample_solutions_data)
#         sol_ic = 1
#         solutions.set_ic(sol_ic)
#         model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)

#         model.set_wd(temp_dir)
#         model.set_database(mock_database)
        
#         stress = ChemStress('WEL-1')
#         stress.set_spd([2])
#         stress.set_packtype('WEL')
        
#         model.set_chem_stress(stress)
#         assert hasattr(model, 'WEL-1')
#         assert getattr(model, 'WEL-1') == stress
    
#     def test_set_chem_stress_invalid_type(self, sample_solutions_data):
#         """Test set_chem_stress with invalid type."""
#         solutions = Solutions(sample_solutions_data)
#         sol_ic = 1
#         solutions.set_ic(sol_ic)
#         model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=3)
        
#         with pytest.raises(AssertionError):
#             model.set_chem_stress("invalid")


# class TestMup3dConfig:
#     """Test suite for Mup3d configuration methods."""
    
#     def test_set_config(self, sample_solutions_data):
#         """Test setting configuration."""
#         solutions = Solutions(sample_solutions_data)
#         model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=10)
        
#         config = model.set_config(
#             reactive_externalio=True,
#             nthreads=4
#         )
#         assert config.reactive_externalio is True
#         assert config.nthreads == 4
    
#     def test_get_config(self, sample_solutions_data):
#         """Test getting configuration."""
#         solutions = Solutions(sample_solutions_data)
#         model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=10)
#         model.set_config(nthreads=2)
        
#         config_dict = model.get_config()
#         assert isinstance(config_dict, dict)
#         assert config_dict['nthreads'] == 2
    
#     def test_save_config(self, sample_solutions_data, temp_dir):
#         """Test saving configuration to file."""
#         solutions = Solutions(sample_solutions_data)
#         model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=10)
#         model.set_wd(temp_dir)
#         model.set_config(nthreads=3)
        
#         config_path = model.save_config()
#         assert os.path.exists(config_path)
#         assert config_path.name == 'mf6rtm.toml'


class TestMup3dInitialConditions:
    """Test suite for initial conditions handling."""

    def test_solutions_none(self, sample_solutions_data):
        """Test error when solutions is None."""
        solutions = None
        
        with pytest.raises(ValueError):
            model = Mup3d(solutions=solutions, nlay=2, nrow=3, ncol=4)
    
    def test_solutions_ic_single_value(self, sample_solutions_data):
        """Test solutions IC with single value."""
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(2)
        model = Mup3d(solutions=solutions, nlay=2, nrow=3, ncol=4)
        
        assert model.solutions.ic.shape == (2, 3, 4)
        assert np.all(model.solutions.ic == 2)
    
    def test_solutions_ic_array(self, sample_solutions_data):
        """Test solutions IC with array."""
        solutions = Solutions(sample_solutions_data)
        ic = np.ones((2, 3, 4), dtype=int)
        ic[0, :, :] = 1
        ic[1, :, :] = 2
        solutions.set_ic(ic)
        
        model = Mup3d(solutions=solutions, nlay=2, nrow=3, ncol=4)
        np.testing.assert_array_equal(model.solutions.ic, ic)


class TestMup3dFileOperations:
    """Test suite for file operations."""
    
    def test_set_postfix(self, sample_solutions_data, temp_dir):
        """Test setting postfix file."""
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(1)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=10)
        
        postfix_file = os.path.join(temp_dir, 'postfix.txt')
        with open(postfix_file, 'w') as f:
            f.write('# Postfix content')
        
        model.set_postfix(postfix_file)
        assert model.postfix == postfix_file
    
    def test_set_postfix_nonexistent(self, sample_solutions_data):
        """Test set_postfix with nonexistent file."""
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(1)
        model = Mup3d(solutions=solutions, nlay=1, nrow=1, ncol=10)
        
        with pytest.raises(AssertionError):
            model.set_postfix('/nonexistent/file.txt')


class TestMup3dSaveLoad:
    """Test suite for save/load operations."""
    
    def test_save_mup3d(self, sample_solutions_data, temp_dir):
        """Test saving Mup3d object."""
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(1)
        model = Mup3d(solutions=solutions, nlay=1, nrow=2, ncol=5)
        model.set_wd(temp_dir)
        model.set_initial_temp(25.0)
        model.set_componenth2o(True)
        
        model.save_mup3d('test_model.pkl')
        
        pkl_file = os.path.join(temp_dir, 'test_model.pkl')
        assert os.path.exists(pkl_file)
    
    def test_load_mup3d(self, sample_solutions_data, temp_dir, mock_database):
        """Test loading Mup3d object."""
        # Create and save a model
        solutions = Solutions(sample_solutions_data)
        solutions.set_ic(1)
        model = Mup3d(solutions=solutions, nlay=1, nrow=2, ncol=5)
        model.set_wd(temp_dir)
        model.set_database(mock_database)
        model.set_initial_temp(30.0)
        model.set_componenth2o(True)
        model.set_charge_offset(5.0)
        
        model.save_mup3d('test_model.pkl')
        
        # Load the model
        loaded_model = Mup3d.load_mup3d('test_model.pkl', temp_dir)
        
        assert loaded_model.nlay == 1
        assert loaded_model.nrow == 2
        assert loaded_model.ncol == 5
        assert loaded_model.init_temp == 30.0
        assert loaded_model.componenth2o is True
        assert loaded_model.charge_offset == 5.0

