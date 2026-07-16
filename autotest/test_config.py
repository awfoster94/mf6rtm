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
