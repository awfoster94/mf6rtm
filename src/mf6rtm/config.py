import toml
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import numpy as np

class MF6RTMConfig:
    """MF6RTM Configuration class similar to FloPy package structure.
    This class provides a FloPy-style interface for configuring MF6RTM
    reaction timing parameters.

    Parameters
    ----------
    reaction_timing : str, optional
        Controls when reactions are calculated. Options:
        - 'all' : Calculate reactions at all time steps (default)
        - 'user' : Calculate reactions only at user-specified time steps
        - 'adaptive' : Use adaptive timing based on convergence criteria
    tsteps : List[Tuple[int, int]], optional
        List of (kper, kstp) tuples specifying when reactions should be calculated.
        Only used when reaction_timing='user'. Default is empty list.
        kper is stress period (1-based), kstp is time step (1-based).

    Attributes
    ----------
    reaction_timing : str
        Current reaction timing strategy.
    tsteps : List[Tuple[int, int]]
        List of time steps for reaction calculations.
    """

    def __init__(self,
                 reactive: bool = True,
                 reaction_timing: str = 'all',
                 tsteps: List[Tuple[int, int]] = None):
        """Initialize MF6RTM configuration.

        Parameters
        ----------
        reaction_timing : str, optional
            Reaction timing strategy ('all', 'user', 'adaptive').
        tsteps : List[Tuple[int, int]], optional
            List of (kper, kstp) tuples for user-defined timing.
        """
        self.reactive = reactive
        self.reaction_timing = reaction_timing
        self.tsteps = tsteps if tsteps is not None else []

        # Validate inputs
        self._validate_reaction_timing()
        self._validate_tsteps()

    def _validate_reaction_timing(self):
        """Validate reaction_timing parameter."""
        valid_options = ['all', 'user', 'adaptive']
        if self.reaction_timing not in valid_options:
            raise ValueError(f"reaction_timing must be one of {valid_options}, "
                           f"got '{self.reaction_timing}'")

    def _validate_tsteps(self):
        """Validate tsteps parameter."""
        if not isinstance(self.tsteps, list):
            raise ValueError("tsteps must be a list")
        normalized = []
        for i, tstep in enumerate(self.tsteps):
            if not isinstance(tstep, (tuple, list)) or len(tstep) != 2:
                raise ValueError(f"tsteps[{i}] must be a tuple/list of length 2")

            kper, kstp = tstep
            if not isinstance(kper, int) or not isinstance(kstp, int):
                raise ValueError(f"tsteps[{i}] must contain integers")
            if kper < 1 or kstp < 1:
                raise ValueError(f"tsteps[{i}]: kper and kstp must be 1-indexed")
            normalized.append((kper, kstp))  # force into tuple
        # Ensure (1, 1) is included
        if (1, 1) not in normalized:
            normalized.insert(0, (1, 1))


    def get_tsteps_for_period(self, kper: int) -> List[int]:
        """Get time steps for a specific stress period.

        Parameters
        ----------
        kper : int
            Stress period number (1-based).

        Returns
        -------
        List[int]
            List of time step numbers for the given stress period.

        Examples
        --------
        >>> config = MF6RTMConfig(reaction_timing='user',
        ...                       tsteps=[(1, 1), (1, 10), (2, 5)])
        >>> config.get_tsteps_for_period(1)
        [1, 10]
        """
        return sorted([kstp for kp, kstp in self.tsteps if kp == kper])

    def is_reaction_tstep(self, kper: int, kstp: int) -> bool:
        """Check if reactions should be calculated at a specific time step.

        Parameters
        ----------
        kper : int
            Stress period number (1-based).
        kstp : int
            Time step number (1-based).

        Returns
        -------
        bool
            True if reactions should be calculated at this time step.

        Examples
        --------
        >>> config = MF6RTMConfig(reaction_timing='user', tsteps=[(1, 1)])
        >>> config.is_reaction_tstep(1, 1)
        True
        >>> config.is_reaction_tstep(1, 2)
        False
        """
        if self.reaction_timing == 'all':
            return True
        elif self.reaction_timing == 'user':
            return (kper, kstp) in self.tsteps
        elif self.reaction_timing == 'adaptive':
            # Placeholder for adaptive
            return True
        else:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for TOML output.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for TOML serialization.
        """
        dict = {
            'global': {
                        'reactive':self.reactive,
            },
            'reaction_timing': {
                'strategy': self.reaction_timing,
                'tsteps': self.tsteps
            }
        }
        return dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MF6RTMConfig':
        """Create configuration from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Dictionary loaded from TOML file.

        Returns
        -------
        MF6RTMConfig
            New configuration instance.

        Raises
        ------
        ValueError
            If required keys are missing or have invalid values.
        """
        if 'reaction_timing' not in config_dict:
            raise ValueError("Missing required 'reaction_timing' block")

        rt_config = config_dict['reaction_timing']

        # Extract strategy
        strategy = rt_config.get('strategy', 'all')

        # Extract tsteps
        tsteps = rt_config.get('tsteps')

        return cls(reaction_timing=strategy, tsteps=tsteps)

    @classmethod
    def from_toml_file(cls, filepath: str) -> 'MF6RTMConfig':
        """Load configuration from TOML file.

        Parameters
        ----------
        filepath : str
            Path to TOML configuration file.

        Returns
        -------
        MF6RTMConfig
            New configuration instance loaded from file.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = toml.load(f)
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        except toml.TomlDecodeError as e:
            raise ValueError(f"Invalid TOML format in {filepath}: {e}")

    def save_to_file(self, filepath: str):
        """Save configuration to TOML file.

        Parameters
        ----------
        filepath : str
            Path where TOML file should be saved.
        """
        config_dict = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            toml.dump(config_dict, f)
        # print(f"Configuration saved to: {filepath}")

    def __repr__(self):
        """String representation of the configuration."""
        return (f"MF6RTMConfig(reaction_timing='{self.reaction_timing}', "
                f"tsteps={self.tsteps})")

    def __str__(self):
        """Detailed string representation."""
        lines = [f"MF6RTM will run with the following configuration:"]
        lines.append(f"  Reactive: {self.reactive}")
        lines.append(f"  Reaction timing: {self.reaction_timing}")

        if self.reaction_timing == 'user' and self.tsteps:
            lines.append(f"  User-defined time steps ({len(self.tsteps)} total):")
            for kper, kstp in sorted(self.tsteps):
                lines.append(f"    Period {kper}, Step {kstp}")
        elif self.reaction_timing == 'all':
            lines.append("  Reactions calculated at all time steps")

        return '\n'.join(lines)
