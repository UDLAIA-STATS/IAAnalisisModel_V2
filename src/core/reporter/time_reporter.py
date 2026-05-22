from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import time
import traceback

import logfire

from src.config.routes import TIME_REPORTS_DIR


class ProcessTimeReporter:
    def __init__(self, match_id: int):
        self.start_times = {}
        self.end_time = 0
        self._create_report_file(match_id)
        self.stats = defaultdict(lambda: {"count": 0, "total": 0.0, "min": float("inf"), "max": 0.0})

    def _create_report_file(self, match_id: int):
        try:
            self.report_file = Path(TIME_REPORTS_DIR, f"time_report_{match_id}.json")
            self.report_file.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "metadata": {
                    "match_id": match_id,
                    "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            }
            self.report_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

            logfire.info(f"[Time Reporter] Archivo de reporte de tiempo creado: {self.report_file}")
        except:
            logfire.error(f"[Time Reporter] Error al crear el archivo de reporte de tiempo: {traceback.format_exc()}")
            raise

    def stop(self, process: str):
        start_time = self.start_times.pop(process, None)

        if start_time is None:
            raise Exception("No se ha iniciado el proceso")

        duration = time.perf_counter() - start_time

        stat = self.stats[process]
        stat["count"] += 1
        stat["total"] += duration
        stat["min"] = min(stat["min"], duration)
        stat["max"] = max(stat["max"], duration)

        logfire.info(f"[Time Reporter] Proceso '{process}' detenido. Duracion: {duration:.4f} segundos")

    def publish(self):
        initial_data = self.report_file.read_text("utf-8")

        report = json.loads(initial_data)
        report["stats"] = dict(self.stats)
        self.report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))

        logfire.info(f"[Time Reporter] Reporte de tiempo publicado en {self.report_file}")

    def start(self, process: str):
        self.start_times[process] = time.perf_counter()
