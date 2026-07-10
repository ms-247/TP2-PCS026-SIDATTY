"""Genere un jeu de donnees synthetique de capteurs climatiques (Parquet).

Simule le contenu de data/raw_sensors/ : mesures geospatiales avec quelques
lignes hors-domaine volontairement injectees pour exercer le filtrage Lazy
de src.numerical_core.ingest_sensor_coordinates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw_sensors" / "sensors.parquet"


def generate(n_valid: int = 500, n_invalid: int = 20, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)

    valid_latitude = rng.uniform(-90.0, 90.0, n_valid)
    valid_longitude = rng.uniform(-180.0, 180.0, n_valid)

    invalid_latitude = rng.uniform(91.0, 120.0, n_invalid)
    invalid_longitude = rng.uniform(181.0, 200.0, n_invalid)

    latitude = np.concatenate([valid_latitude, invalid_latitude])
    longitude = np.concatenate([valid_longitude, invalid_longitude])
    temperature_celsius = rng.normal(15.0, 8.0, n_valid + n_invalid)
    sensor_id = np.arange(n_valid + n_invalid)

    return pl.DataFrame(
        {
            "sensor_id": sensor_id,
            "latitude": latitude,
            "longitude": longitude,
            "temperature_celsius": temperature_celsius,
        }
    )


def main() -> None:
    frame = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUTPUT_PATH)
    print(f"Wrote {frame.height} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
