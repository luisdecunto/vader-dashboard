from __future__ import annotations

import csv
import math
import random
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
COLUMNS = [
    "time_from_onset_s",
    "vertical_distance_L",
    "radial_Hencky_strain",
    "vertical_strain",
    "D_over_D0",
    "force_g",
    "diameter_mm",
]


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    rng = random.Random(20260723)

    materials = {
        "PDMS_soft": {"modulus": 55.0, "relaxation": 0.018, "diameter": 1.65},
        "xanthan_gel_batch_A": {"modulus": 92.0, "relaxation": 0.026, "diameter": 1.48},
        "protein_mix_v2": {"modulus": 72.0, "relaxation": 0.021, "diameter": 1.55},
    }
    velocities = {"2mms": 2.0, "5mms": 5.0, "10mms": 10.0}
    samples = ["S01", "S02"]

    for material, material_params in materials.items():
        for velocity_label, velocity in velocities.items():
            for sample_index, sample in enumerate(samples):
                rows = make_rows(
                    rng=rng,
                    material_params=material_params,
                    velocity=velocity,
                    sample_index=sample_index,
                )
                filename = f"{material}_{velocity_label}_{sample}_PROC1_FAST_processed.csv"
                write_csv(DATA_DIR / filename, rows)


def make_rows(
    rng: random.Random,
    material_params: dict[str, float],
    velocity: float,
    sample_index: int,
) -> list[dict[str, float]]:
    row_count = 700
    dt = 0.045
    initial_length_mm = 7.5 + rng.uniform(-0.3, 0.3)
    initial_diameter_mm = material_params["diameter"] + rng.uniform(-0.04, 0.04)
    modulus = material_params["modulus"] * (1.0 + rng.uniform(-0.08, 0.08))
    relaxation = material_params["relaxation"]
    sample_shift = 1.0 + sample_index * 0.035
    rows: list[dict[str, float]] = []

    for i in range(row_count):
        time_s = i * dt
        vertical_distance_l = (velocity * time_s / initial_length_mm) * 0.09
        vertical_strain = math.log1p(vertical_distance_l * sample_shift)

        progress = i / (row_count - 1)
        radial_noise = rng.gauss(0, 0.035)
        radial_hencky = max(0.0, 1.0 + 9.0 * progress + radial_noise)
        d_over_d0 = max(
            0.18,
            math.exp(-0.16 * radial_hencky) + rng.gauss(0, 0.002),
        )
        diameter_mm = initial_diameter_mm * d_over_d0 + rng.gauss(0, 0.004)

        elastic_force = modulus * (1.0 - math.exp(-3.2 * vertical_strain))
        viscous_force = relaxation * velocity * 55.0
        oscillation = 1.4 * math.sin(time_s * 1.7 + sample_index)
        noise = rng.gauss(0, 0.55)
        force_g = elastic_force + viscous_force + oscillation + noise

        necking_start = 0.58 + rng.uniform(-0.015, 0.015)
        if vertical_strain > necking_start:
            force_g *= max(0.25, 1.0 - (vertical_strain - necking_start) * 1.8)

        rows.append(
            {
                "time_from_onset_s": round(time_s, 4),
                "vertical_distance_L": round(vertical_distance_l, 6),
                "radial_Hencky_strain": round(radial_hencky, 6),
                "vertical_strain": round(vertical_strain, 6),
                "D_over_D0": round(d_over_d0, 6),
                "force_g": round(force_g, 6),
                "diameter_mm": round(diameter_mm, 6),
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
