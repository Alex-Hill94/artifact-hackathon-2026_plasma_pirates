"""CLARA S04 section, knobs and screen readout, shared by the three pipelines.

Everything here comes from the organisers' public files in challenges/optimisation:
the Cheetah lattice, the archiver magnet ranges, the reference tune, the
current-to-strength conversion (utils.py) and the kickoff notebook's section,
screen, starting beam and target. Nothing is read from the machine.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import cheetah
import numpy as np
import torch

CHALLENGE = Path(__file__).resolve().parents[4] / "challenges" / "optimisation"
LATTICE_DIR = CHALLENGE / "data" / "clara" / "lattice_data"
LATTICE_JSON = LATTICE_DIR / "CLARA_cheetah.json"

# Kickoff notebook, section 4: the Easy objective.
LATTICE_START = "cla_s01_sim_aper_01"
SECTION_START = "cla_s04_sim_aper_01"
SECTION_END = "cla_s04_sim_mark_02"
SCREEN = "cla_s04_dia_scr_03"
SCREEN_PV = "CLA-S04-DIA-SCR-03"
ENTRANCE_ENERGY = 193.156e6  # eV at the S04 entrance
NORMALISED_EMITTANCE = 1e-6  # m rad
ELECTRON_REST_ENERGY = 0.51099895e6  # eV
NUM_PARTICLES = 10_000
TARGET_UM = np.array([0.0, 0.0, 100.0, 100.0])  # mu_x, mu_y, sigma_x, sigma_y

# score.py: below this fraction of the beam on the camera chip there is no reading.
MIN_SCREEN_FRACTION = 0.10

# (PV, Cheetah element, property). The real machine has separate H and V corrector
# supplies; the lattice models each pair as one CombinedCorrector.
KNOBS = [(f"CLA-S04-MAG-QUAD-0{i}:SETI", f"cla_s04_mag_quad_0{i}", "k1") for i in (1, 2, 3, 4)] + [
    (f"CLA-S04-MAG-{plane}COR-0{i}:SETI", f"cla_s04_mag_hvcor_0{i}", prop)
    for i in (1, 2)
    for plane, prop in (("H", "horizontal_angle"), ("V", "vertical_angle"))
]
LABELS = ["mu_x_um", "mu_y_um", "sigma_x_um", "sigma_y_um"]

_spec = importlib.util.spec_from_file_location("clara_lattice_utils", LATTICE_DIR / "utils.py")
utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(utils)
MOMENTUM_MEVC = utils.beam_energy["s04"]  # what utils.update_cheetah_element converts with


def column(pv: str) -> str:
    """CLA-S04-MAG-QUAD-01:SETI -> cla_s04_mag_quad_01_seti (feature names are lower snake case)."""
    return pv.lower().replace("-", "_").replace(":", "_")


FEATURES = [column(pv) for pv, _, _ in KNOBS]


def lattice_hash() -> str:
    return hashlib.sha1(LATTICE_JSON.read_bytes()).hexdigest()[:12]


def _elements(name: str) -> dict:
    return json.loads((LATTICE_DIR / name).read_text())["elements"]


def section() -> cheetah.Segment:
    """S04 up to SCR-03, at the reference tune. Partitions share elements with the parent."""
    lattice = cheetah.Segment.from_lattice_json(str(LATTICE_JSON)).partition_at(LATTICE_START, mode="after")[-1]
    s04 = lattice.partition_at(SECTION_START, mode="before")[-1].partition_at(SECTION_END, mode="after")[0]
    segment = s04.partition_at(SCREEN, mode="after")[0]
    for name, props in _elements("reference_tune.json").items():
        if name in segment.element_names:
            for prop, value in props.items():
                setattr(getattr(segment, name), prop, torch.tensor(float(value)))
    # Moments are taken from the particles; the simulated camera image is not needed.
    getattr(segment, SCREEN).is_active = False
    return segment


def _to_current(pv: str, element: str, prop: str, value: float) -> float:
    if prop == "k1":
        return utils.quadrupole_current_from_k(element, value, MOMENTUM_MEVC)
    corrector = pv.split(":")[0]
    return utils.corrector_current_from_angle(corrector, value, MOMENTUM_MEVC)


def reference_currents() -> dict:
    """The reference tune as SETI currents, A."""
    tune = _elements("reference_tune.json")
    return {column(pv): _to_current(pv, el, prop, tune[el][prop]) for pv, el, prop in KNOBS}


def bounds(spread: float = 1.0) -> dict:
    """Per-knob current range, A: reference +/- spread x half the archiver range, clipped to it."""
    ranges, tune = _elements("magnet_ranges.json"), _elements("reference_tune.json")
    out = {}
    for pv, el, prop in KNOBS:
        low, high = ranges[el][prop]
        ref, half = tune[el][prop], (high - low) / 2
        lo, hi = max(low, ref - spread * half), min(high, ref + spread * half)
        a, b = _to_current(pv, el, prop, lo), _to_current(pv, el, prop, hi)
        out[column(pv)] = (min(a, b), max(a, b))
    return out


def starting_beam(seed: int) -> cheetah.ParticleBeam:
    """Kickoff notebook sample_beam(seed), with the particle draw seeded too."""
    rng = np.random.default_rng(seed)
    beta_x, alpha_x = rng.uniform(8.0, 16.0), rng.uniform(-1.0, 1.0)
    beta_y, alpha_y = rng.uniform(8.0, 16.0), rng.uniform(-1.0, 1.0)
    mu = [rng.uniform(-3e-4, 3e-4), rng.uniform(-1e-4, 1e-4), rng.uniform(-3e-4, 3e-4), rng.uniform(-1e-4, 1e-4)]
    eps = NORMALISED_EMITTANCE / (ENTRANCE_ENERGY / ELECTRON_REST_ENERGY)
    torch.manual_seed(seed)
    return cheetah.ParticleBeam.from_parameters(
        num_particles=NUM_PARTICLES,
        energy=torch.tensor(ENTRANCE_ENERGY),
        mu_x=torch.tensor(mu[0]), mu_px=torch.tensor(mu[1]),
        mu_y=torch.tensor(mu[2]), mu_py=torch.tensor(mu[3]),
        sigma_x=torch.tensor(np.sqrt(eps * beta_x)),
        sigma_px=torch.tensor(np.sqrt(eps * (1 + alpha_x**2) / beta_x)),
        cov_xpx=torch.tensor(-eps * alpha_x),
        sigma_y=torch.tensor(np.sqrt(eps * beta_y)),
        sigma_py=torch.tensor(np.sqrt(eps * (1 + alpha_y**2) / beta_y)),
        cov_ypy=torch.tensor(-eps * alpha_y),
        sigma_tau=torch.tensor(1e-4),
        sigma_p=torch.tensor(1e-3),
        dtype=torch.float32,
    )


def capture(segment, beam, currents: dict, noise_fraction: float = 0.0, rng=None) -> dict:
    """Set the SETI currents, track, read SCR-03 the way score.py's camera does.

    Moments are over the part of the beam that lands on the chip, relative to the
    screen centre. Below MIN_SCREEN_FRACTION there is no reading (NaN moments).
    """
    for pv, _, _ in KNOBS:
        utils.update_cheetah_element(segment, pv, currents[column(pv)])
    out = segment.track(beam)
    x, y = out.x.detach(), out.y.detach()
    extent = getattr(segment, SCREEN).extent
    on_chip = (out.survival_probabilities.detach() > 0.5) & (x.abs() <= float(extent[1])) & (y.abs() <= float(extent[3]))
    fraction = float(on_chip.sum()) / beam.num_particles
    lost = fraction < MIN_SCREEN_FRACTION
    moments = np.full(4, np.nan) if lost else np.array(
        [x[on_chip].mean().item(), y[on_chip].mean().item(), x[on_chip].std().item(), y[on_chip].std().item()]
    ) * 1e6
    if noise_fraction and not lost:
        moments = moments + rng.normal(0.0, noise_fraction * moments[2:].mean(), size=4)
    return dict(zip(LABELS, moments.tolist())) | {"screen_fraction": fraction, "beam_lost": lost}


def objective_um(moments) -> float:
    """Kickoff objective: mean |moments - target|, here in um."""
    return float(np.abs(np.asarray(moments) - TARGET_UM).mean())
