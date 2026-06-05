"""
SHAP-style Explainability for Pricing Recommendations
Explains WHY each price recommendation was made.
"""

import numpy as np
import pandas as pd


class PricingExplainer:
    """Explain pricing recommendations in human-readable format."""

    def __init__(self):
        self.feature_names = [
            "cost",
            "demand_score",
            "competition_level",
            "quality_score",
            "season_factor",
            "ga_adjustment",
        ]
        self.feature_labels = [
            "Cost Base",
            "Demand Effect",
            "Competition Impact",
            "Quality Premium",
            "Seasonal Adjustment",
            "GA Optimisation",
        ]

    # ── Public API ────────────────────────────────────────────────────────
    def explain_price(self, product_data: dict, recommended_price: float) -> dict:
        """
        Generate a full explanation for a single product's recommended price.

        Args:
            product_data:      dict with product features
            recommended_price: the price output by the pipeline

        Returns:
            explanation dict ready to pass to explain.html Jinja context
        """
        factors, factor_values = self._compute_waterfall(product_data, recommended_price)
        radar_labels, radar_values = self._compute_radar(product_data)
        fired_rules = self._get_fired_rules(product_data)

        margin = 0.0
        cost = product_data.get("cost", 0)
        if cost and recommended_price:
            margin = (recommended_price - cost) / recommended_price * 100

        return {
            "product_name":       product_data.get("product_name", "Unknown"),
            "recommended_price":  round(recommended_price, 2),
            "cost":               round(cost, 2),
            "profit_margin":      round(margin, 2),
            "confidence":         round(product_data.get("confidence_score", 75), 1),
            "demand_score":       product_data.get("demand_score", 50),
            "competition_level":  product_data.get("competition_level", 50),
            # Waterfall
            "factors":            factors,
            "factor_values":      factor_values,
            # Radar
            "radar_labels":       radar_labels,
            "radar_values":       radar_values,
            # Decision path
            "fired_rules":        fired_rules,
            # Text explanation
            "text_explanation":   self._text_explanation(product_data, recommended_price),
        }

    def explain_batch(self, df: pd.DataFrame) -> list:
        """Explain all products in a results DataFrame."""
        explanations = []
        price_col = "recommended_price" if "recommended_price" in df.columns else "final_price"
        for _, row in df.iterrows():
            product_data = row.to_dict()
            price = product_data.get(price_col, product_data.get("cost", 0) * 1.3)
            explanations.append(self.explain_price(product_data, price))
        return explanations

    def get_summary_stats(self, explanations: list) -> dict:
        """Aggregate stats across all explanations."""
        if not explanations:
            return {}
        margins     = [e["profit_margin"]  for e in explanations]
        confidences = [e["confidence"]     for e in explanations]
        prices      = [e["recommended_price"] for e in explanations]
        return {
            "avg_margin":     round(np.mean(margins),     2),
            "avg_confidence": round(np.mean(confidences), 2),
            "avg_price":      round(np.mean(prices),      2),
            "min_price":      round(np.min(prices),       2),
            "max_price":      round(np.max(prices),       2),
            "high_conf_pct":  round(np.mean([c > 75 for c in confidences]) * 100, 1),
        }

    # ── Internal helpers ─────────────────────────────────────────────────
    def _compute_waterfall(self, product_data: dict, final_price: float):
        """Compute waterfall contributions summing to final_price."""
        cost       = float(product_data.get("cost", 0))
        demand     = float(product_data.get("demand_score", 50))
        comp       = float(product_data.get("competition_level", 50))
        quality    = float(product_data.get("quality_score", 5))
        season_raw = product_data.get("season", "medium")

        # Map season to numeric
        season_map = {"low": -1, "medium": 0, "high": 1}
        season_val = season_map.get(str(season_raw).lower(), 0)

        # Contribution estimates
        cost_contrib    = cost
        demand_contrib  = (demand - 50) / 100 * cost * 0.4
        comp_contrib    = -(comp - 50)  / 100 * cost * 0.2
        quality_contrib = (quality - 5) / 10  * cost * 0.15
        season_contrib  = season_val * cost * 0.08
        ga_contrib      = final_price - (cost_contrib + demand_contrib + comp_contrib +
                                         quality_contrib + season_contrib)

        factors = self.feature_labels
        values  = [
            round(cost_contrib,    2),
            round(demand_contrib,  2),
            round(comp_contrib,    2),
            round(quality_contrib, 2),
            round(season_contrib,  2),
            round(ga_contrib,      2),
        ]
        return factors, values

    def _compute_radar(self, product_data: dict):
        """Compute 0–100 scores for radar chart axes."""
        labels = ["Demand Strength", "Competition Pressure", "Quality Index",
                  "Season Advantage", "Cost Efficiency"]
        cost  = float(product_data.get("cost",              1))
        price = float(product_data.get("recommended_price", cost * 1.3))

        demand_score    = float(product_data.get("demand_score",       50))
        comp_score      = 100 - float(product_data.get("competition_level", 50))  # lower comp → higher score
        quality_raw     = float(product_data.get("quality_score",       5))
        quality_score   = quality_raw / 10 * 100

        season_map     = {"low": 20, "medium": 50, "high": 85}
        season_score   = season_map.get(str(product_data.get("season", "medium")).lower(), 50)

        cost_efficiency = min(100, max(0, (price - cost) / max(cost, 0.01) * 200))

        values = [
            round(demand_score,    1),
            round(comp_score,      1),
            round(quality_score,   1),
            round(season_score,    1),
            round(cost_efficiency, 1),
        ]
        return labels, values

    def _get_fired_rules(self, product_data: dict) -> list:
        """Simulate fuzzy rule activations for display."""
        demand = float(product_data.get("demand_score",       50))
        comp   = float(product_data.get("competition_level",  50))
        qual   = float(product_data.get("quality_score",      5))

        # Linguistic assignments
        demand_label = "HIGH" if demand > 66 else ("MEDIUM" if demand > 33 else "LOW")
        comp_label   = "HIGH" if comp   > 66 else ("MEDIUM" if comp   > 33 else "LOW")
        qual_label   = "HIGH" if qual   > 7  else ("MEDIUM" if qual   > 4  else "LOW")

        rules = [
            {
                "condition": f"IF demand IS {demand_label} AND competition IS {comp_label}",
                "output":    "PREMIUM" if (demand > 60 and comp < 50) else
                             "BUDGET"  if (demand < 40 and comp > 60) else "FAIR",
                "activation": round(min(demand / 100, 1 - comp / 100) + 0.1, 2),
            },
            {
                "condition": f"IF quality IS {qual_label} AND demand IS {demand_label}",
                "output":    "PREMIUM" if qual > 7 else ("FAIR" if qual > 4 else "BUDGET"),
                "activation": round(qual / 10 * 0.8 + 0.1, 2),
            },
            {
                "condition": f"IF competition IS {comp_label}",
                "output":    "BUDGET" if comp > 66 else ("FAIR" if comp > 33 else "PREMIUM"),
                "activation": round(abs(comp - 50) / 100 + 0.2, 2),
            },
        ]
        # Sort by activation descending
        return sorted(rules, key=lambda r: r["activation"], reverse=True)

    def _text_explanation(self, product_data: dict, price: float) -> str:
        """One-paragraph natural language explanation."""
        cost   = product_data.get("cost", 0)
        demand = product_data.get("demand_score", 50)
        comp   = product_data.get("competition_level", 50)
        name   = product_data.get("product_name", "This product")

        demand_label = "strong" if demand > 66 else ("moderate" if demand > 33 else "weak")
        comp_label   = "intense" if comp > 66 else ("moderate" if comp > 33 else "low")
        margin_pct   = (price - cost) / max(cost, 0.01) * 100

        return (
            f"{name} is recommended at ₹{price:.2f} based on a cost of ₹{cost:.2f} "
            f"({margin_pct:.1f}% margin). Market demand is **{demand_label}** and "
            f"competition is **{comp_label}**, which the fuzzy inference engine used "
            f"to calibrate the price. The genetic algorithm then fine-tuned the parameters "
            f"to maximise the profit–satisfaction objective."
        )