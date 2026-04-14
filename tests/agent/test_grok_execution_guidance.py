"""Tests for Grok-specific execution guidance injection."""

import pytest
from agent.prompt_builder import GROK_EXECUTION_GUIDANCE


class TestGrokExecutionGuidance:
    """Verify GROK_EXECUTION_GUIDANCE constant structure and content."""

    def test_is_nonempty_string(self):
        assert isinstance(GROK_EXECUTION_GUIDANCE, str)
        assert len(GROK_EXECUTION_GUIDANCE) > 100

    def test_contains_no_intent_phrases_block(self):
        assert "<no_intent_phrases>" in GROK_EXECUTION_GUIDANCE
        assert "</no_intent_phrases>" in GROK_EXECUTION_GUIDANCE

    def test_contains_execute_first_block(self):
        assert "<execute_first>" in GROK_EXECUTION_GUIDANCE
        assert "</execute_first>" in GROK_EXECUTION_GUIDANCE

    def test_contains_no_analysis_hallucination_block(self):
        assert "<no_analysis_hallucination>" in GROK_EXECUTION_GUIDANCE
        assert "</no_analysis_hallucination>" in GROK_EXECUTION_GUIDANCE

    def test_prohibits_intent_phrases(self):
        text = GROK_EXECUTION_GUIDANCE.lower()
        assert "i will" in text or "let me" in text  # mentions phrases to avoid
        assert "tool" in text  # mentions calling tools instead
