
import sqlite3
from pathlib import Path


class MicroCache:

    PRESSURE_GRAD_RTOL = 1e-3
    PRESSURE_GRAD_ATOL = 1

    APERTURE_RTOL = 1e-3
    APERTURE_ATOL = 1e-5

    def __init__(self, database_path):

        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    def _connect(self):

        conn = sqlite3.connect(
            self.database_path,
        )

        return conn

    def _initialize_database(self):
        with self._connect() as conn:

            # Create the complete table for a new database.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS micro_results (
                    pressure_gradient REAL,
                    mean_aperture REAL,
                    scalar_solver_active INTEGER,
                    flux REAL,
                    dispersion REAL,
                    counter INTRGER
                )
            """)

    def lookup(self, inputs):

        dp = float(inputs["pressure_gradient"])
        aperture = float(inputs["mean_aperture"])
        scalar_active = int(
            inputs["scalar_solver_active"]
        )

        dp_min = dp / (1.0 + self.PRESSURE_GRAD_RTOL)  - self.PRESSURE_GRAD_ATOL
        dp_max = dp / (1.0 - self.PRESSURE_GRAD_RTOL) + self.PRESSURE_GRAD_ATOL
        aperture_min = aperture / (1.0 + self.APERTURE_RTOL) - self.APERTURE_ATOL
        aperture_max = aperture / (1.0 - self.APERTURE_RTOL)+ self.APERTURE_ATOL

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    rowid,
                    pressure_gradient,
                    flux,
                    dispersion

                FROM micro_results

                WHERE scalar_solver_active = ?
                  AND mean_aperture BETWEEN ? AND ?
                  AND pressure_gradient BETWEEN ? AND ?
                """,
                (
                    scalar_active,
                    aperture_min,
                    aperture_max,
                    dp_min,
                    dp_max,
                ),
            ).fetchone()
        if row:
            rowid, cached_dp, cached_flux, cached_dispersion = row
            conn.execute(
                "UPDATE micro_results SET counter = counter + 1 WHERE rowid = ?",
                (rowid,))
            conn.commit()
            return cached_flux / cached_dp * dp, cached_dispersion
        return None, None

    def store(self, inputs, result):
        with self._connect() as conn:

            conn.execute(
                """
                INSERT INTO micro_results (
                    pressure_gradient,
                    mean_aperture,
                    scalar_solver_active,
                    flux,
                    dispersion,
                    counter
                )

                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (
                    float(inputs["pressure_gradient"]),
                    float(inputs["mean_aperture"]),
                    int(inputs["scalar_solver_active"]),
                    float(result["flux"]),
                    float(result["dispersion"])
                ),
            )
