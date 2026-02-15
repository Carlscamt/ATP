"""
Health Checks & Alerts

Daily automated checks for model health, data freshness,
GPU availability, and performance degradation.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger("monitor")


class HealthChecker:
    """Run daily health checks and flag issues."""

    def __init__(self):
        self.checks: list[dict[str, Any]] = []

    def run_all(self) -> list[dict[str, Any]]:
        """Run all health checks."""
        self.checks = []
        self._check_gpu()
        self._check_model_exists()
        self._check_data_freshness()
        self._check_scraper_health()
        self._check_performance()

        # Summary
        failed = [c for c in self.checks if c["status"] == "FAIL"]
        warnings = [c for c in self.checks if c["status"] == "WARNING"]

        if failed:
            logger.error("HEALTH CHECK FAILED: %d failures", len(failed))
            for f in failed:
                logger.error("  ❌ %s: %s", f["check"], f["message"])
        elif warnings:
            logger.warning("HEALTH CHECK: %d warnings", len(warnings))
        else:
            logger.info("HEALTH CHECK: All checks passed ✅")

        return self.checks

    def _add(self, check: str, status: str, message: str):
        self.checks.append({
            "check": check,
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })

    def _check_gpu(self):
        """Verify NVIDIA GPU is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                self._add("GPU", "PASS", f"GPU available: {gpu_info}")
            else:
                self._add("GPU", "FAIL", "nvidia-smi failed")
        except FileNotFoundError:
            self._add("GPU", "FAIL", "nvidia-smi not found — no GPU?")
        except Exception as e:
            self._add("GPU", "WARNING", f"GPU check error: {e}")

    def _check_model_exists(self):
        """Verify trained model file exists."""
        model_path = Path("model/xgboost_model.json")
        if model_path.exists():
            age_hours = (datetime.now().timestamp() - model_path.stat().st_mtime) / 3600
            if age_hours > 168:  # > 1 week
                self._add("Model", "WARNING", f"Model is {age_hours:.0f}h old — retrain?")
            else:
                self._add("Model", "PASS", f"Model exists ({age_hours:.0f}h old)")
        else:
            self._add("Model", "FAIL", "No model file found at model/xgboost_model.json")

    def _check_data_freshness(self):
        """Verify recent data exists."""
        paths_to_check = [
            ("Raw Matches", "data/raw/matches.parquet"),
            ("Features", "data/processed/features.parquet"),
        ]

        for name, path in paths_to_check:
            p = Path(path)
            if p.exists():
                age_hours = (datetime.now().timestamp() - p.stat().st_mtime) / 3600
                if age_hours > 48:
                    self._add(f"Data ({name})", "WARNING", f"{age_hours:.0f}h old")
                else:
                    self._add(f"Data ({name})", "PASS", f"Fresh ({age_hours:.0f}h old)")
            else:
                self._add(f"Data ({name})", "FAIL", f"Not found: {path}")

    def _check_scraper_health(self):
        """Check if scrapers ran recently."""
        upcoming_path = Path("data/upcoming/matches.parquet")
        if upcoming_path.exists():
            try:
                df = pl.read_parquet(upcoming_path)
                if "scraped_at" in df.columns:
                    latest = df["scraped_at"].max()
                    self._add("Scraper", "PASS", f"Last run: {latest}")
                else:
                    self._add("Scraper", "PASS", "Upcoming data exists")
            except Exception as e:
                self._add("Scraper", "WARNING", f"Error reading upcoming data: {e}")
        else:
            self._add("Scraper", "WARNING", "No upcoming match data found")

    def _check_performance(self):
        """Check recent betting performance for degradation."""
        log_path = Path("data/predictions/bet_log.parquet")
        if not log_path.exists():
            self._add("Performance", "WARNING", "No bet log found")
            return

        try:
            df = pl.read_parquet(log_path)
            if "profit" not in df.columns or len(df) < 10:
                self._add("Performance", "WARNING", "Insufficient bet history")
                return

            # Check last 30 bets
            recent = df.tail(30)
            roi = recent["profit"].sum() / recent["stake"].sum() * 100 if "stake" in df.columns else 0

            if roi < -10:
                self._add("Performance", "FAIL", f"ROI={roi:.1f}% — severe underperformance!")
            elif roi < 0:
                self._add("Performance", "WARNING", f"ROI={roi:.1f}% — negative (last 30 bets)")
            else:
                self._add("Performance", "PASS", f"ROI={roi:.1f}% (last 30 bets)")
        except Exception as e:
            self._add("Performance", "WARNING", f"Error checking performance: {e}")


# CLI
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    checker = HealthChecker()
    results = checker.run_all()

    print("\n=== HEALTH CHECK REPORT ===")
    for r in results:
        icon = "✅" if r["status"] == "PASS" else "⚠️" if r["status"] == "WARNING" else "❌"
        print(f"  {icon} {r['check']}: {r['message']}")
