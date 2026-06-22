"""Configuration objects for the PyLEADER package.

These dataclasses replace the hard-coded "top cell" of the original notebooks
(``LEADER_python_final.ipynb``, ``..._bg``, ``..._forcedN`` and
``make_LEADER_obs_files.ipynb``).  Defaults reproduce the notebook values so
that running with no overrides matches the historical workflow; the CLI scripts
in ``scripts/`` expose every field as a command-line argument.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default location of the WISE/NEOWISE working data on this machine.  The
# notebooks were executed with this as the working directory, so the input
# catalog files (``neowise_mainbelt.csv``, ``AllMBAFamilyMembers.txt``) and all
# ``Fam*_data_*`` / ``*_analysis_*`` directories live here.
DEFAULT_BASE_DIR = "/Users/ssonnett/Desktop/work/MBA_SFDs"


@dataclass
class AnalysisConfig:
    """Configuration for a LEADER shape/spin inversion run.

    A single ``AnalysisConfig`` replaces the three near-identical analysis
    notebooks.  The notebook variants map onto config flags:

    * ``LEADER_python_final``     -> ``population_kind="family"``, ``forced_n=False``
    * ``LEADER_python_final_bg``  -> ``population_kind="background"``, ``forced_n=False``
    * ``LEADER_python_forcedN``   -> ``forced_n=True`` (subsample to ``wanted`` points)
    """

    # --- core sample selection (from the notebook top cell) ---
    famid: str = "3815"                     # family / population to explore
    cat: str = "allsky_4band_p1bs_psd"      # catalog used to generate the .obs files
    filterpriority: str = "w3"              # photometry filter to analyze
    diam_low: float = 5.0                   # lower diameter limit of sample (km)
    diam_high: float = 10.0                 # upper diameter limit of sample (km)

    # --- statistics / tolerances ---
    phase_angle_limit: float = 40.0         # upper solar phase-angle limit (deg)
    Ndraws: int = 1000                      # random draws per trial
    Ntrials: int = 100                      # number of trials (repeats of the experiment)
    date_tol: float = 60.0                  # max JD gap between points in one apparition
    wanted: int = 5                         # min data points per object per epoch

    # --- behaviour flags ---
    overwrite: bool = False                 # overwrite (vs. append/skip) existing output
    convert2degrees: bool = True            # report/plot beta in degrees

    # --- input catalog file (relative to base_dir or absolute) ---
    neowise_fle: str = "neowise_mainbelt.csv"

    # --- variant selectors (collapse the three notebooks) ---
    population_kind: str = "family"         # "family" -> Fam<famid>; "background" -> <famid>
    forced_n: bool = False                  # forcedN: subsample each object to `wanted` amplitudes

    base_dir: str = DEFAULT_BASE_DIR

    def __post_init__(self) -> None:
        if self.population_kind not in ("family", "background"):
            raise ValueError(
                "population_kind must be 'family' or 'background', "
                f"got {self.population_kind!r}"
            )

    @property
    def _famtoken(self) -> str:
        """Family token used in directory names.

        Mirrors the notebook difference: ``LEADER_python_final`` prefixes the
        numeric family id with ``Fam``; ``LEADER_python_final_bg`` uses the
        (already descriptive) population id verbatim.
        """
        return f"Fam{self.famid}" if self.population_kind == "family" else self.famid

    @property
    def datadir(self) -> str:
        """Directory of input ``.obs`` files (matches the notebook's ``datadir``)."""
        return (
            f"{self.base_dir}/{self._famtoken}_data_"
            f"{self.cat}_{self.filterpriority}/"
        )

    @property
    def outdir(self) -> str:
        """Output analysis directory (matches the notebook's ``outdir``).

        The ``forcedN`` notebook prefixes the directory with ``ForcedN<wanted>_``.
        """
        prefix = f"ForcedN{self.wanted}_" if self.forced_n else ""
        return (
            f"{self.base_dir}/{prefix}{self._famtoken}_analysis_"
            f"{self.cat}_{self.filterpriority}_"
            f"{self.diam_low}km_to_{self.diam_high}km"
        )

    @property
    def neowise_path(self) -> str:
        """Absolute path to the NEOWISE catalog file.

        Absolute paths are used as-is; bare filenames are resolved against
        ``base_dir`` (where the notebooks found them).
        """
        if self.neowise_fle.startswith("/"):
            return self.neowise_fle
        return f"{self.base_dir}/{self.neowise_fle}"


@dataclass
class ObsBuildConfig:
    """Configuration for building LEADER ``.obs`` input files from IRSA/Horizons.

    Replaces the top cell of ``make_LEADER_obs_files.ipynb``.
    """

    famid: str = "350"                       # collisional family identifier
    cat: str = "allsky_4band_p1bs_psd"       # IRSA catalog to query
    min_obs: int = 5                         # min observations to write a .obs file
    istart: int = 0                          # index to resume from (after interruption)
    overwrite: bool = False                  # overwrite existing data dir / curl script
    filterpriority: str = "w3"               # filter to analyze (lowercase)

    family_file: str = "AllMBAFamilyMembers.txt"  # MBA family membership listing
    neowise_fle: str = "neowise_mainbelt.csv"     # NEOWISE-determined properties (PDS SBN)

    base_dir: str = DEFAULT_BASE_DIR

    @property
    def data_dir(self) -> str:
        """Directory the ``.obs`` files are written to (notebook: ``Fam<id>_data_...``)."""
        return f"{self.base_dir}/Fam{self.famid}_data_{self.cat}_{self.filterpriority}"

    @property
    def ifilt(self) -> int:
        """Index into the cc_flags / ph_qual strings for the chosen filter."""
        if self.filterpriority == "w2":
            return 1
        if self.filterpriority == "w3":
            return 2
        raise ValueError(f"Unsupported filterpriority {self.filterpriority!r}")

    @property
    def family_path(self) -> str:
        if self.family_file.startswith("/"):
            return self.family_file
        return f"{self.base_dir}/{self.family_file}"

    @property
    def neowise_path(self) -> str:
        if self.neowise_fle.startswith("/"):
            return self.neowise_fle
        return f"{self.base_dir}/{self.neowise_fle}"

    @property
    def curl_script(self) -> str:
        """Path of the optional curl download script (notebook: ``GetWiseData_FamID*.sh``)."""
        return f"{self.base_dir}/GetWiseData_FamID{self.famid}.sh"
