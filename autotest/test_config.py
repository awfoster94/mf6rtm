import pytest
from mf6rtm.config.config import MF6RTMConfig


class TestMF6RTMConfigSolverSection:
    """Tests for the [solver] section of MF6RTMConfig."""

    def test_default_min_concentration_is_none(self):
        """solver.min_concentration defaults to None when not in config."""
        cfg = MF6RTMConfig()
        assert cfg.solver['min_concentration'] is None

    def test_from_dict_reads_min_concentration(self):
        """from_dict reads min_concentration from the [solver] section."""
        cfg = MF6RTMConfig.from_dict({'solver': {'min_concentration': 1e-6}})
        assert cfg.solver['min_concentration'] == pytest.approx(1e-6)

    def test_from_dict_defaults_when_solver_absent(self):
        """from_dict sets min_concentration to None when [solver] is absent."""
        cfg = MF6RTMConfig.from_dict({})
        assert cfg.solver['min_concentration'] is None

    def test_from_dict_defaults_when_key_absent(self):
        """from_dict defaults min_concentration to None when key missing from [solver]."""
        cfg = MF6RTMConfig.from_dict({'solver': {}})
        assert cfg.solver['min_concentration'] is None

    def test_to_dict_omits_solver_when_all_none(self):
        """to_dict excludes [solver] entirely when min_concentration is None (TOML has no null)."""
        cfg = MF6RTMConfig()
        d = cfg.to_dict()
        assert 'solver' not in d

    def test_to_dict_includes_solver_when_set(self):
        """to_dict includes [solver] when min_concentration has a value."""
        cfg = MF6RTMConfig()
        cfg.solver['min_concentration'] = 1e-30
        d = cfg.to_dict()
        assert 'solver' in d
        assert d['solver']['min_concentration'] == pytest.approx(1e-30)

    def test_round_trip_via_toml(self, tmp_path):
        """Saving and reloading a config preserves min_concentration."""
        import toml
        cfg = MF6RTMConfig()
        cfg.solver['min_concentration'] = 1e-8
        filepath = tmp_path / "mf6rtm.toml"
        cfg.save_to_file(str(filepath))

        reloaded = MF6RTMConfig.from_toml_file(str(filepath))
        assert reloaded.solver['min_concentration'] == pytest.approx(1e-8)

    def test_round_trip_none_omitted_from_toml(self, tmp_path):
        """When min_concentration is None, the saved TOML has no [solver] section."""
        import toml
        cfg = MF6RTMConfig()
        filepath = tmp_path / "mf6rtm.toml"
        cfg.save_to_file(str(filepath))

        with open(filepath) as f:
            raw = toml.load(f)
        assert 'solver' not in raw


class TestMF6RTMConfigStr:
    """Tests for MF6RTMConfig.__str__ output."""

    def test_str_shows_no_clipping_when_none(self):
        """__str__ reports 'no clipping' when min_concentration is None."""
        cfg = MF6RTMConfig()
        text = str(cfg)
        assert 'no clipping' in text.lower()

    def test_str_shows_value_when_set(self):
        """__str__ shows the mol/L value when min_concentration is set."""
        cfg = MF6RTMConfig()
        cfg.solver['min_concentration'] = 1e-6
        text = str(cfg)
        assert 'mol/L' in text
        assert '1.00e-06' in text

    def test_str_lists_user_timesteps(self):
        """__str__ enumerates each (period, step) when timing is 'user'."""
        cfg = MF6RTMConfig(reactive={'timing': 'user', 'tsteps': [(1, 1), (2, 3)]})
        text = str(cfg)
        assert 'User-defined time steps' in text
        assert 'Period 1, Step 1' in text
        assert 'Period 2, Step 3' in text

    def test_repr_contains_class_and_attrs(self):
        """__repr__ renders the class name and the reactive attribute."""
        cfg = MF6RTMConfig()
        text = repr(cfg)
        assert text.startswith('MF6RTMConfig(')
        assert 'reactive=' in text


class TestMF6RTMConfigValidateTsteps:
    """Tests for _validate_tsteps branch handling."""

    def test_user_timing_empty_tsteps_raises(self):
        cfg = MF6RTMConfig(reactive={'timing': 'user', 'tsteps': []})
        with pytest.raises(ValueError):
            cfg._validate_tsteps()

    def test_all_timing_with_tsteps_warns(self, capsys):
        cfg = MF6RTMConfig(reactive={'timing': 'all', 'tsteps': [(1, 1)]})
        cfg._validate_tsteps()
        assert 'WARNING' in capsys.readouterr().out

    def test_bad_tstep_shape_raises(self):
        cfg = MF6RTMConfig(reactive={'timing': 'user', 'tsteps': [(1, 1, 1)]})
        with pytest.raises(ValueError):
            cfg._validate_tsteps()

    def test_non_integer_tstep_raises(self):
        cfg = MF6RTMConfig(reactive={'timing': 'user', 'tsteps': [(1.0, 2.0)]})
        with pytest.raises(ValueError):
            cfg._validate_tsteps()

    def test_non_one_indexed_tstep_raises(self):
        cfg = MF6RTMConfig(reactive={'timing': 'user', 'tsteps': [(0, 1)]})
        with pytest.raises(ValueError):
            cfg._validate_tsteps()


class TestMF6RTMConfigOutputSection:
    """Tests for the [output] section round-trip."""

    def test_default_output_format_is_csv(self):
        cfg = MF6RTMConfig()
        assert cfg.output['output_format'] == 'csv'

    def test_from_dict_reads_output_format(self):
        cfg = MF6RTMConfig.from_dict({'output': {'output_format': 'hdf5'}})
        assert cfg.output['output_format'] == 'hdf5'

    def test_to_dict_includes_output(self):
        cfg = MF6RTMConfig()
        d = cfg.to_dict()
        assert d['output']['output_format'] == 'csv'

    def test_round_trip_via_toml(self, tmp_path):
        cfg = MF6RTMConfig()
        cfg.output['output_format'] = 'hdf5'
        filepath = tmp_path / "mf6rtm.toml"
        cfg.save_to_file(str(filepath))

        reloaded = MF6RTMConfig.from_toml_file(str(filepath))
        assert reloaded.output['output_format'] == 'hdf5'


class TestMF6RTMConfigFromTomlErrors:
    """Tests for from_toml_file error handling."""

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            MF6RTMConfig.from_toml_file('/nonexistent/path/mf6rtm.toml')

    def test_invalid_toml_raises_valueerror(self, tmp_path):
        filepath = tmp_path / "bad.toml"
        filepath.write_text("this is = = not valid toml [[[")
        with pytest.raises(ValueError):
            MF6RTMConfig.from_toml_file(str(filepath))
