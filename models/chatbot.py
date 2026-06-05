"""
AI Pricing Assistant Chatbot
Rule-based + fuzzy logic for natural language pricing explanations
"""

import re
import json
import numpy as np


class PricingChatbot:
    """Intelligent chatbot for pricing explanations"""

    def __init__(self):
        self.context = {}
        self.conversation_history = []
        self.intents = self._build_intents()

    # ── Intent / knowledge base ─────────────────────────────────────────
    def _build_intents(self):
        return [
            {
                "name": "fuzzy_logic",
                "keywords": ["fuzzy", "fuzzy logic", "mamdani", "membership", "inference", "linguistic"],
                "response": (
                    "**Fuzzy Logic** is the core pricing intelligence 🧠\n\n"
                    "Instead of hard thresholds, it uses *linguistic variables* like:\n"
                    "- **Low / Medium / High** demand\n"
                    "- **Weak / Moderate / Strong** competition\n"
                    "- **Budget / Fair / Premium** price\n\n"
                    "The Mamdani inference engine fires 18+ rules simultaneously and "
                    "combines them via centroid defuzzification. This handles market *uncertainty* naturally."
                )
            },
            {
                "name": "genetic_algorithm",
                "keywords": ["genetic", "ga", "genetic algorithm", "evolution", "chromosome", "mutation",
                             "crossover", "optimize", "optimise", "parameter"],
                "response": (
                    "**Genetic Algorithm** fine-tunes 8 pricing parameters 🧬\n\n"
                    "1. Encodes parameters as a *chromosome* (float array)\n"
                    "2. Runs a *population* of 50–100 candidate solutions\n"
                    "3. Evaluates each via fitness = profit_weight × profit + satisfaction_weight × satisfaction\n"
                    "4. Selects best (tournament selection)\n"
                    "5. Applies *crossover* + *mutation* to breed new solutions\n"
                    "6. Repeats for ~100 generations\n\n"
                    "Result: near-optimal parameters without brute-force search."
                )
            },
            {
                "name": "confidence_score",
                "keywords": ["confidence", "confidence score", "score", "low score", "trust"],
                "response": (
                    "**Confidence Score** measures recommendation reliability 📊\n\n"
                    "It drops when:\n"
                    "- Input data has *missing values* or outliers\n"
                    "- Fuzzy rule activation is weak (inputs near boundary)\n"
                    "- Features are highly contradictory\n"
                    "- Insufficient rows for GA convergence (<10 rows)\n\n"
                    "**Improve it:** upload cleaner data, add ≥50 rows, and ensure "
                    "cost / demand / competition columns are filled."
                )
            },
            {
                "name": "profit",
                "keywords": ["profit", "margin", "revenue", "income", "improve profit", "increase profit"],
                "response": (
                    "**Improving profit margin** 💰\n\n"
                    "Key levers:\n"
                    "- ↑ GA `profit_weight` parameter (in genetic_algorithm.py)\n"
                    "- Tag rows with `season = high` to activate premium multiplier (~1.15×)\n"
                    "- Low competition → system auto-recommends premium pricing\n"
                    "- Reduce unit cost → shifts fuzzy membership → triggers higher-margin rules\n\n"
                    "Re-run optimization after any dataset change."
                )
            },
            {
                "name": "dataset_format",
                "keywords": ["upload", "csv", "dataset", "file", "format", "column", "data"],
                "response": (
                    "**Expected CSV format** 📁\n\n"
                    "Required columns:\n"
                    "- `product_name` — string\n"
                    "- `cost` — unit cost (₹ or $)\n"
                    "- `demand_score` — 0–100\n"
                    "- `competition_level` — 0–100\n\n"
                    "Optional (auto-imputed if missing):\n"
                    "- `quality_score` — 0–10\n"
                    "- `season` — low / medium / high\n"
                    "- `current_price` — for comparison\n\n"
                    "Minimum **10 rows** recommended for meaningful GA convergence."
                )
            },
            {
                "name": "season",
                "keywords": ["season", "seasonal", "factor", "multiplier", "peak", "off-season"],
                "response": (
                    "**Seasonal adjustment** 🌦️\n\n"
                    "- `high` season → multiplier 1.10–1.25 (premium opportunity)\n"
                    "- `medium` season → multiplier ~1.0 (neutral)\n"
                    "- `low` season → multiplier 0.85–0.95 (price cut for volume)\n\n"
                    "Tag your rows with a `season` column for automatic adjustment."
                )
            },
            {
                "name": "explainer",
                "keywords": ["explain", "why", "reason", "shap", "waterfall", "radar", "decision", "factor"],
                "response": (
                    "**Price Explanation** on the Results page 💡\n\n"
                    "- **Waterfall chart** — shows how each factor (cost, demand, competition, season) "
                    "contributes to the final price\n"
                    "- **Radar chart** — feature importance across 5 dimensions\n"
                    "- **Decision path** — which fuzzy rules fired and their activation strength\n\n"
                    "This is a SHAP-style post-hoc explanation built on fuzzy activation weights."
                )
            },
            {
                "name": "dashboard",
                "keywords": ["dashboard", "chart", "visualization", "graph", "plot", "scatter", "histogram"],
                "response": (
                    "**Dashboard views** 📈\n\n"
                    "1. **Price Distribution** — histogram of recommended vs original prices\n"
                    "2. **Profit vs Satisfaction** — interactive scatter coloured by confidence\n"
                    "3. **GA Convergence** — fitness over generations (confirms algorithm converged)\n\n"
                    "All charts use Plotly — you can zoom, pan, and export as PNG."
                )
            },
        ]

    # ── Public API ──────────────────────────────────────────────────────
    def get_response(self, user_message: str, context: dict = None) -> dict:
        """
        Process a user message and return a response dict.

        Returns:
            {
                "response": str,
                "intent":   str,
                "confidence": float,
                "suggestions": list[str]
            }
        """
        if context:
            self.context.update(context)

        msg_lower = user_message.lower().strip()
        self.conversation_history.append({"role": "user", "content": user_message})

        # Greeting
        if re.search(r'\b(hi|hello|hey|good morning|good evening|howdy)\b', msg_lower):
            resp = self._greeting_response()
            return self._build_result(resp, "greeting", 1.0)

        # Thanks
        if re.search(r'\b(thanks|thank you|thx|cheers)\b', msg_lower):
            resp = "You're welcome! 😊 Let me know if you have more questions about your pricing data."
            return self._build_result(resp, "thanks", 1.0)

        # Keyword matching
        best_intent = None
        best_score  = 0.0
        for intent in self.intents:
            score = self._intent_score(msg_lower, intent["keywords"])
            if score > best_score:
                best_score  = score
                best_intent = intent

        if best_intent and best_score > 0:
            resp = best_intent["response"]
            # Personalise with context if available
            resp = self._personalise(resp, best_intent["name"])
            self.conversation_history.append({"role": "assistant", "content": resp})
            return self._build_result(resp, best_intent["name"], best_score)

        # Fallback
        fallback = self._fallback_response(user_message)
        self.conversation_history.append({"role": "assistant", "content": fallback})
        return self._build_result(fallback, "unknown", 0.0)

    def get_price_explanation(self, product_data: dict) -> str:
        """Generate natural-language explanation for a specific product."""
        name   = product_data.get("product_name", "this product")
        price  = product_data.get("recommended_price", 0)
        cost   = product_data.get("cost", 0)
        margin = product_data.get("profit_margin", 0)
        conf   = product_data.get("confidence", 0)
        demand = product_data.get("demand_score", 50)
        comp   = product_data.get("competition_level", 50)

        demand_label = "high" if demand > 66 else ("medium" if demand > 33 else "low")
        comp_label   = "strong" if comp > 66 else ("moderate" if comp > 33 else "weak")

        explanation = (
            f"**{name}** — Recommended price: ₹{price:.2f}\n\n"
            f"**Why this price?**\n"
            f"- Base cost is ₹{cost:.2f}, giving a **{margin:.1f}% profit margin**\n"
            f"- Market demand is **{demand_label}** ({demand:.0f}/100)\n"
            f"- Competition is **{comp_label}** ({comp:.0f}/100)\n"
            f"- Overall confidence: **{conf:.0f}%**\n\n"
        )

        if demand > 66:
            explanation += "💡 *High demand allows a premium price without hurting volume.*\n"
        elif demand < 33:
            explanation += "💡 *Low demand suggests a competitive price to attract buyers.*\n"

        if comp < 33:
            explanation += "💡 *Weak competition gives room for higher margins.*\n"
        elif comp > 66:
            explanation += "💡 *Strong competition warrants a sharper price.*\n"

        return explanation

    # ── Helpers ─────────────────────────────────────────────────────────
    def _intent_score(self, text: str, keywords: list) -> float:
        matches = sum(1 for kw in keywords if kw in text)
        return matches / max(len(keywords), 1)

    def _personalise(self, response: str, intent_name: str) -> str:
        """Inject context-specific data into generic responses."""
        if intent_name == "confidence_score" and "last_confidence" in self.context:
            c = self.context["last_confidence"]
            level = "excellent" if c > 80 else ("acceptable" if c > 60 else "low")
            response += f"\n\n*Your last run had a {level} confidence of {c:.0f}%.*"
        if intent_name == "profit" and "last_margin" in self.context:
            m = self.context["last_margin"]
            response += f"\n\n*Your current average margin is {m:.1f}%.*"
        return response

    def _greeting_response(self) -> str:
        return (
            "👋 Hi! I'm the **FuzzyPrice AI Assistant**.\n\n"
            "I can help you understand:\n"
            "- How **fuzzy logic** determines prices\n"
            "- What the **genetic algorithm** optimises\n"
            "- How to read **confidence scores**\n"
            "- Tips to **improve profit margins**\n"
            "- How to **format your CSV** dataset\n\n"
            "What would you like to know?"
        )

    def _fallback_response(self, user_message: str) -> str:
        suggestions = [
            "fuzzy logic pricing rules",
            "genetic algorithm parameters",
            "confidence scores",
            "CSV dataset format",
            "seasonal adjustments",
        ]
        return (
            f"I'm not sure about *\"{user_message[:60]}\"*, but I can help with:\n\n"
            + "\n".join(f"- {s}" for s in suggestions)
            + "\n\nTry asking about one of these topics!"
        )

    @staticmethod
    def _build_result(response: str, intent: str, confidence: float) -> dict:
        suggestions_map = {
            "fuzzy_logic":      ["genetic algorithm", "confidence scores"],
            "genetic_algorithm": ["fuzzy logic", "profit improvement"],
            "confidence_score": ["dataset format", "fuzzy logic"],
            "profit":           ["seasonal factors", "confidence score"],
            "dataset_format":   ["upload tips", "confidence score"],
            "unknown":          ["fuzzy logic", "genetic algorithm", "confidence score"],
        }
        return {
            "response":    response,
            "intent":      intent,
            "confidence":  round(confidence, 2),
            "suggestions": suggestions_map.get(intent, []),
        }

    def update_context(self, key: str, value):
        self.context[key] = value

    def reset(self):
        self.context = {}
        self.conversation_history = []