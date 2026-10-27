"""
Reproduction tests for evaluator execution on ContextGraph decision recording.
"""

from unittest.mock import Mock
import pytest

from semantica.context.agent_context import AgentContext
from semantica.context.context_graph import ContextGraph


class TestContextGraphDecisionEvaluatorsRepro:
    @pytest.fixture
    def mock_vector_store(self):
        store = Mock()
        store.add = Mock()
        store.search = Mock(return_value=[])
        return store

    def test_record_decision_with_evaluators_attaches_metadata(self, mock_vector_store):
        """ContextGraph decision recording must run evaluators and attach eval metadata."""
        graph = ContextGraph()
        context = AgentContext(
            vector_store=mock_vector_store,
            knowledge_graph=graph,
            decision_tracking=True,
            evaluators=["decision_scores"],
            eval_config={
                "decision_scores": {
                    "expected_outcome": "approved",
                    "min_confidence": 0.8,
                }
            },
        )

        decision_id = context.record_decision(
            category="credit_approval",
            scenario="Credit limit increase request",
            reasoning="Customer has excellent payment history and low debt",
            outcome="approved",
            confidence=0.9,
            decision_maker="ai_agent_001",
            cross_system_context={"salesforce": {"tier": "gold"}},
        )

        assert decision_id is not None
        assert decision_id in graph.nodes
        node = graph.nodes[decision_id]

        assert "eval_score" in node.properties, f"eval_score missing from node properties: {node.properties}"
        assert node.properties["eval_score"] is not None
        assert "eval_passed" in node.properties, f"eval_passed missing from node properties: {node.properties}"
        assert node.properties["eval_passed"] is True
        assert "eval_details" in node.properties, f"eval_details missing from node properties: {node.properties}"
        assert "cross_system_context" in node.properties

    def test_record_decision_evaluator_failure_non_blocking(self, mock_vector_store):
        """Failing evaluator must not prevent decision persistence."""
        graph = ContextGraph()

        def broken_evaluator(decision, **kwargs):
            raise RuntimeError("Evaluator execution crash")

        context = AgentContext(
            vector_store=mock_vector_store,
            knowledge_graph=graph,
            decision_tracking=True,
            evaluators=[broken_evaluator],
        )

        decision_id = context.record_decision(
            category="credit_approval",
            scenario="Credit limit increase request",
            reasoning="Customer has excellent payment history",
            outcome="approved",
            confidence=0.9,
        )

        assert decision_id is not None
        assert decision_id in graph.nodes
        node = graph.nodes[decision_id]
        assert "eval_score" in node.properties
        assert node.properties["eval_passed"] is False

    def test_record_decision_without_evaluators_unchanged(self, mock_vector_store):
        """Omission of evaluators preserves existing behavior."""
        graph = ContextGraph()
        context = AgentContext(
            vector_store=mock_vector_store,
            knowledge_graph=graph,
            decision_tracking=True,
        )

        decision_id = context.record_decision(
            category="credit_approval",
            scenario="Credit limit increase request",
            reasoning="Customer has excellent payment history",
            outcome="approved",
            confidence=0.9,
        )

        assert decision_id is not None
        assert decision_id in graph.nodes
        node = graph.nodes[decision_id]
        assert "eval_score" not in node.properties
        assert "eval_passed" not in node.properties
        assert "eval_details" not in node.properties
