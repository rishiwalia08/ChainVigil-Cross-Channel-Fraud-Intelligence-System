"""
ChainVigil — LLM Plain-English Explainer (Hugging Face)

Creates regulator-friendly, human-readable account explanations from:
  - model score
  - top feature attributions
  - existing XAI reasoning

If HF inference is unavailable, gracefully falls back to deterministic text.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

import requests


class LLMExplainer:
    """Generate plain-English fraud explanations using Hugging Face Inference API."""

    def __init__(self):
        self.model = os.getenv("CHAINVIGIL_HF_MODEL", "google/flan-t5-small")
        self.token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
        self.timeout_seconds = float(os.getenv("CHAINVIGIL_HF_TIMEOUT", "18"))
        self.url = f"https://api-inference.huggingface.co/models/{self.model}"

    def summarize_account(
        self,
        account_id: str,
        confidence_score: float,
        feature_attributions: List[Dict],
        base_reasoning: str,
    ) -> Dict:
        """
        Returns:
          {
            "summary": str,
            "meta": {"source": str, "model": str}
          }
        """
        prompt = self._build_prompt(
            account_id=account_id,
            confidence_score=confidence_score,
            feature_attributions=feature_attributions,
            base_reasoning=base_reasoning,
        )

        llm_text = self._query_hf(prompt)
        if llm_text:
            return {
                "summary": llm_text,
                "meta": {
                    "source": "huggingface-api",
                    "model": self.model,
                },
            }

        return {
            "summary": self._fallback_summary(
                account_id=account_id,
                confidence_score=confidence_score,
                feature_attributions=feature_attributions,
                base_reasoning=base_reasoning,
            ),
            "meta": {
                "source": "template-fallback",
                "model": self.model,
            },
        }

    def _build_prompt(
        self,
        account_id: str,
        confidence_score: float,
        feature_attributions: List[Dict],
        base_reasoning: str,
    ) -> str:
        top = feature_attributions[:5]
        top_lines = []
        for feat in top:
            name = str(feat.get("name", "unknown")).replace("_", " ")
            weight = float(feat.get("importance", 0.0)) * 100
            top_lines.append(f"- {name}: {weight:.1f}%")

        return (
            "You are an AML/fraud analyst assistant. "
            "Write a concise plain-English explanation for a suspicious account. "
            "Keep it under 110 words. Include: (1) risk summary, "
            "(2) strongest 2-3 drivers, (3) immediate action recommendation. "
            "Avoid legal claims and avoid mentioning model internals. "
            "Start each sentence with a capital letter.\n\n"
            f"Account: {account_id}\n"
            f"Risk score: {confidence_score * 100:.1f}%\n"
            f"Top drivers:\n{chr(10).join(top_lines) if top_lines else '- none'}\n"
            f"Base reasoning: {base_reasoning}\n"
        )

    def _query_hf(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 160,
                "temperature": 0.2,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }

        try:
            resp = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            if resp.status_code >= 400:
                return ""

            data = resp.json()
            if isinstance(data, list) and data:
                txt = data[0].get("generated_text", "")
                return self._clean(txt)
            if isinstance(data, dict):
                if "generated_text" in data:
                    return self._clean(data.get("generated_text", ""))
                # some models return {"error": "..."}
                return ""
            return ""
        except Exception:
            return ""

    @staticmethod
    def _clean(text: str) -> str:
        cleaned = " ".join((text or "").strip().split())
        if not cleaned:
            return ""

        # Normalize spacing and sentence starts.
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        fixed = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if p[0].isalpha():
                p = p[0].upper() + p[1:]
            fixed.append(p)

        out = " ".join(fixed)
        if out.lower().startswith("summary:"):
            out = "Summary:" + out[len("summary:"):]
        elif not out.startswith("Summary:"):
            out = f"Summary: {out}"

        if out and out[-1] not in ".!?":
            out += "."
        return out

    def _fallback_summary(
        self,
        account_id: str,
        confidence_score: float,
        feature_attributions: List[Dict],
        base_reasoning: str,
    ) -> str:
        top = feature_attributions[:3]
        drivers = ", ".join(
            str(f.get("name", "unknown")).replace("_", " ")
            for f in top
        ) or "network behavior signals"

        if confidence_score >= 0.85:
            action = "apply temporary restrictions and start urgent review"
        elif confidence_score >= 0.60:
            action = "trigger enhanced due diligence and close monitoring"
        else:
            action = "monitor with periodic analyst checks"

        return (
            f"Summary: Account {account_id} is flagged at {confidence_score * 100:.1f}% risk. "
            f"Main drivers are {drivers}. "
            f"Recommended next step: {action}. "
            f"Context: {base_reasoning}"
        )
