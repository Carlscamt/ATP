"""
Betting Strategy — Fractional Kelly Criterion & Expected Value

Calculates optimal stake sizing with bankroll protection.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("betting")


def calculate_ev(model_prob: float, decimal_odds: float) -> float:
    """
    Expected Value for a bet.
    EV = (p * (odds - 1)) - (1 - p)

    Returns 0 if negative EV.
    """
    ev = (model_prob * (decimal_odds - 1)) - (1 - model_prob)
    return max(ev, 0.0)


def full_kelly(model_prob: float, decimal_odds: float) -> float:
    """
    Full Kelly Criterion stake as fraction of bankroll.
    f* = (bp - q) / b
    where b = odds - 1, p = win prob, q = 1 - p
    """
    b = decimal_odds - 1
    if b <= 0:
        return 0.0
    q = 1 - model_prob
    f = (b * model_prob - q) / b
    return max(f, 0.0)


def fractional_kelly(
    model_prob: float,
    decimal_odds: float,
    *,
    fraction: float = 0.25,
    max_stake_pct: float = 0.05,
) -> float:
    """
    Fractional Kelly with stake cap.

    fraction=0.25 → bet 25% of full Kelly recommendation
    max_stake_pct=0.05 → never risk more than 5% of bankroll
    """
    f = full_kelly(model_prob, decimal_odds) * fraction
    return min(f, max_stake_pct)


class BettingStrategy:
    """Production betting strategy with risk management."""

    def __init__(
        self,
        bankroll: float = 10000.0,
        kelly_fraction: float = 0.25,
        max_stake_pct: float = 0.05,
        min_ev: float = 0.0,
        min_odds: float = 1.5,
        max_odds: float = 10.0,
    ):
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct = max_stake_pct
        self.min_ev = min_ev
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.bet_history: list[dict[str, Any]] = []

    def evaluate_bet(
        self,
        match_id: str,
        player_name: str,
        model_prob: float,
        decimal_odds: float,
    ) -> dict[str, Any] | None:
        """
        Evaluate a single betting opportunity.
        Returns bet recommendation or None if no bet.
        """
        # Odds filter
        if decimal_odds < self.min_odds or decimal_odds > self.max_odds:
            return None

        # EV check
        ev = calculate_ev(model_prob, decimal_odds)
        if ev <= self.min_ev:
            return None

        # Stake sizing
        stake_pct = fractional_kelly(
            model_prob, decimal_odds,
            fraction=self.kelly_fraction,
            max_stake_pct=self.max_stake_pct,
        )

        if stake_pct <= 0:
            return None

        stake = self.bankroll * stake_pct

        bet = {
            "match_id": match_id,
            "player_name": player_name,
            "model_prob": round(model_prob, 4),
            "decimal_odds": decimal_odds,
            "implied_prob": round(1 / decimal_odds, 4),
            "expected_value": round(ev, 4),
            "kelly_fraction": round(stake_pct, 4),
            "stake": round(stake, 2),
            "bankroll": round(self.bankroll, 2),
        }

        # Flag high-EV for manual review
        if ev > 0.20:
            logger.warning(
                "HIGH EV BET (%.1f%%): %s @ %.2f — review for overconfidence",
                ev * 100, player_name, decimal_odds,
            )
            bet["flag"] = "HIGH_EV_REVIEW"

        return bet

    def record_result(self, match_id: str, won: bool, stake: float, odds: float):
        """Update bankroll after bet result."""
        if won:
            profit = stake * (odds - 1)
            self.bankroll += profit
        else:
            self.bankroll -= stake

        self.bet_history.append({
            "match_id": match_id,
            "won": won,
            "stake": stake,
            "odds": odds,
            "profit": stake * (odds - 1) if won else -stake,
            "bankroll_after": self.bankroll,
        })

    def get_stats(self) -> dict[str, float]:
        """Compute betting performance statistics."""
        if not self.bet_history:
            return {}

        profits = [b["profit"] for b in self.bet_history]
        wins = sum(1 for b in self.bet_history if b["won"])
        total = len(self.bet_history)
        total_staked = sum(b["stake"] for b in self.bet_history)
        total_profit = sum(profits)

        return {
            "total_bets": total,
            "wins": wins,
            "win_rate": wins / total if total > 0 else 0,
            "total_profit": round(total_profit, 2),
            "roi": round(total_profit / total_staked * 100, 2) if total_staked > 0 else 0,
            "bankroll": round(self.bankroll, 2),
        }
