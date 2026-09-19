"""Exercise the notebook's real graphs with deterministic, offline dependencies.

Run from the repository root: uv run python -m unittest discover -s tests -v
"""

import ast
import asyncio
import contextlib
import inspect
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nbformat
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/03_agentic_adaptive_rag.ipynb"
CELLS = json.loads(NOTEBOOK.read_text())["cells"]


def unexpected_model_call(_):
    raise AssertionError("A test must provide a deterministic model response")


def bootstrap():
    """Supply inert models so the actual prompt chains can be constructed offline."""
    model = RunnableLambda(unexpected_model_call)
    structured_model = SimpleNamespace(with_structured_output=lambda schema: model)
    return {
        "analysis_llm": structured_model,
        "evaluation_llm": structured_model,
        "generation_llm": model,
    }


def load_notebook():
    namespace = bootstrap()
    for index, cell in enumerate(CELLS, 1):
        tags = set(cell["metadata"].get("tags", []))
        if tags & {"imports", "core", "prompts"}:
            exec(compile("".join(cell["source"]), f"{NOTEBOOK.name}:cell {index}", "exec"), namespace)
    return namespace


class FakeChain:
    """Validate actual prompt inputs, record calls, then supply a fake response."""

    def __init__(self, prompt, respond):
        self.prompt = prompt
        self.respond = respond
        self.calls = []

    async def ainvoke(self, inputs):
        self.prompt.invoke(inputs)
        self.calls.append(dict(inputs))
        response = self.respond(inputs)
        return await response if inspect.isawaitable(response) else response

    async def abatch(self, inputs):
        return await asyncio.gather(*(self.ainvoke(item) for item in inputs))


class FakeStore:
    def __init__(self, source):
        self.source = source
        self.calls = []
        self.empty = False

    def similarity_search(self, query, k):
        self.calls.append((query, k))
        if self.empty:
            return []
        return [Document(
            page_content=f"Evidence from {self.source} for {query}",
            metadata={"source": f"{self.source}.csv"},
        )]


def install_fakes(namespace):
    """Only replace IO boundaries; the notebook builds and runs both graphs."""
    ns = namespace
    ns["catalog_store"] = FakeStore("catalog")
    ns["faq_store"] = FakeStore("faq")
    ns["troubleshooting_store"] = FakeStore("troubleshooting")
    ns["routing_chain"] = FakeChain(ns["routing_prompt"], lambda inputs: SimpleNamespace(
        source="catalog", reasoning="Fixture source",
    ))
    ns["rewrite_chain"] = FakeChain(ns["rewrite_prompt"], lambda inputs: SimpleNamespace(
        rewritten_query=f"Improved: {inputs['search_query']}", improvements="Fixture rewrite",
    ))
    ns["grading_chain"] = FakeChain(ns["grading_prompt"], lambda inputs: SimpleNamespace(
        relevant="yes", reasoning="Fixture evidence",
    ))
    ns["subquery_answer_chain"] = FakeChain(ns["subquery_answer_prompt"], lambda inputs: SimpleNamespace(
        content=f"Sub-answer for {inputs['query']}",
    ))
    ns["query_analysis_chain"] = FakeChain(ns["query_analysis_prompt"], lambda inputs: ns["QueryAnalysis"](
        sub_queries=[inputs["query"]], execution_plan="parallel", reasoning="One question",
    ))
    ns["final_answer_chain"] = FakeChain(ns["final_answer_prompt"], lambda inputs: SimpleNamespace(
        content="Final answer based on retrieved evidence",
    ))

    async def web_search(inputs):
        return {"results": [{
            "content": f"Web evidence for {inputs['query']}",
            "url": "https://example.test/support",
            "title": "Support fixture",
        }]}

    ns["tavily_search"] = SimpleNamespace(ainvoke=web_search)
    return ns


class AdaptiveRAGTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"})
        tracing.start()
        self.addCleanup(tracing.stop)
        self.ns = install_fakes(load_notebook())

    def test_notebook_schema_and_cell_syntax(self):
        nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))
        for index, cell in enumerate(CELLS, 1):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"cell {index}", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

    async def test_all_four_retrieval_sources_reach_grading_and_answering(self):
        for source in ("catalog", "faq", "troubleshooting", "web_search"):
            with self.subTest(source=source):
                self.ns["routing_chain"].respond = lambda inputs: SimpleNamespace(source=source, reasoning="Fixture")
                result = await self.ns["run_subquery"]("A focused question")
                self.assertEqual(result["source"], source)
                self.assertEqual(result["retrieval_attempts"], 1)
                self.assertEqual(len(result["documents"]), 1)
                self.assertIn("Grade: 1/1 relevant", result["trace"])
        self.assertEqual(self.ns["rewrite_chain"].calls, [])
        self.assertEqual(result["documents"][0].metadata["source"], "https://example.test/support")

    async def test_retry_rewrites_and_routes_again_then_recovers(self):
        self.ns["routing_chain"].respond = lambda inputs: SimpleNamespace(
            source="faq" if inputs["query"].startswith("Improved:") else "catalog", reasoning="Fixture",
        )
        self.ns["grading_chain"].respond = lambda inputs: SimpleNamespace(
            relevant="yes" if "Evidence from faq" in inputs["document"] else "no",
        )
        result = await self.ns["run_subquery"]("What is the return policy?")
        self.assertEqual(result["retrieval_attempts"], 2)
        self.assertEqual(result["source"], "faq")
        self.assertEqual(result["question"], "What is the return policy?")
        self.assertEqual(len(self.ns["rewrite_chain"].calls), 1)
        self.assertTrue(self.ns["rewrite_chain"].calls[0]["retry_feedback"])
        self.assertIn("faq.csv", self.ns["subquery_answer_chain"].calls[0]["context"])

    async def test_empty_retrieval_stops_at_limit_and_exposes_missing_evidence(self):
        self.ns["catalog_store"].empty = True
        result = await self.ns["run_subquery"]("Unknown product")
        self.assertEqual(result["retrieval_attempts"], 2)
        self.assertEqual(result["documents"], [])
        self.assertEqual(len(self.ns["rewrite_chain"].calls), 1)
        self.assertEqual(self.ns["grading_chain"].calls, [])
        self.assertEqual(self.ns["subquery_answer_chain"].calls[0]["context"], "No relevant documents found.")

    async def test_all_documents_rejected_does_not_retry_forever(self):
        self.ns["grading_chain"].respond = lambda inputs: SimpleNamespace(relevant="no")
        result = await asyncio.wait_for(self.ns["run_subquery"]("Unanswerable question"), timeout=3)
        self.assertEqual(result["retrieval_attempts"], 2)
        self.assertEqual(result["documents"], [])
        self.assertEqual(len(self.ns["grading_chain"].calls), 2)

    async def test_one_attempt_configuration_disables_retrieval_retry(self):
        self.ns["MAX_RETRIEVAL_ATTEMPTS"] = 1
        self.ns["grading_chain"].respond = lambda inputs: SimpleNamespace(relevant="no")
        result = await self.ns["run_subquery"]("Unanswerable question")
        self.assertEqual(result["retrieval_attempts"], 1)
        self.assertEqual(self.ns["rewrite_chain"].calls, [])

    async def test_sequential_dependency_context_survives_a_retry(self):
        first = "Choose a mouse"
        second = "Does it have known issues?"
        self.ns["query_analysis_chain"].respond = lambda inputs: SimpleNamespace(
            sub_queries=[first, second], execution_plan="sequential", reasoning="Dependent questions",
        )
        self.ns["subquery_answer_chain"].respond = lambda inputs: SimpleNamespace(
            content="Choose the GlideMaster MX mouse" if inputs["query"] == first else "Known issues answer",
        )
        self.ns["rewrite_chain"].respond = lambda inputs: SimpleNamespace(
            rewritten_query="GlideMaster MX troubleshooting" if inputs["retry_feedback"] else "GlideMaster MX issues",
        )
        self.ns["grading_chain"].respond = lambda inputs: SimpleNamespace(
            relevant="no" if inputs["query"] == "GlideMaster MX issues" else "yes",
        )

        result = await self.ns["main_graph"].ainvoke({"question": "Choose a mouse and check its issues"})
        initial, retry = self.ns["rewrite_chain"].calls
        self.assertIn("GlideMaster MX", initial["previous_context"])
        self.assertEqual(initial["previous_context"], retry["previous_context"])
        self.assertEqual(initial["question"], second)
        self.assertEqual(retry["question"], second)
        self.assertEqual(retry["search_query"], "GlideMaster MX issues")
        self.assertFalse(initial["retry_feedback"])
        self.assertTrue(retry["retry_feedback"])
        self.assertEqual([r["retrieval_attempts"] for r in result["results"]], [1, 2])
        final_inputs = self.ns["final_answer_chain"].calls[0]
        self.assertEqual(final_inputs["execution_plan"], "sequential")
        self.assertIn("Choose the GlideMaster MX mouse", final_inputs["subquery_results"])
        self.assertIn("Search: GlideMaster MX troubleshooting", final_inputs["subquery_results"])

    async def test_parallel_execution_overlaps_and_keeps_results_and_traces_ordered(self):
        second_answered = asyncio.Event()
        completion_order = []
        self.ns["query_analysis_chain"].respond = lambda inputs: SimpleNamespace(
            sub_queries=["first", "second"], execution_plan="parallel", reasoning="Independent questions",
        )

        async def route(inputs):
            if inputs["query"] == "first":
                await second_answered.wait()
            return SimpleNamespace(source="catalog" if inputs["query"] == "first" else "faq", reasoning="Fixture")

        def answer(inputs):
            completion_order.append(inputs["query"])
            if inputs["query"] == "second":
                second_answered.set()
            return SimpleNamespace(content=f"Answer for {inputs['query']}")

        self.ns["routing_chain"].respond = route
        self.ns["subquery_answer_chain"].respond = answer
        result = await asyncio.wait_for(self.ns["main_graph"].ainvoke({"question": "Two questions"}), timeout=3)
        self.assertEqual(completion_order, ["second", "first"])
        self.assertEqual([r["question"] for r in result["results"]], ["first", "second"])
        first, second = result["results"]
        self.assertIsNot(first["trace"], second["trace"])
        self.assertIn("Route: catalog (Fixture)", first["trace"])
        self.assertNotIn("Route: faq (Fixture)", first["trace"])
        final_context = self.ns["final_answer_chain"].calls[0]["context"]
        self.assertIn("Evidence from catalog for first", final_context)
        self.assertIn("Evidence from faq for second", final_context)
        self.assertNotIn("Answer for", final_context)

    async def test_single_parent_query_and_repeated_runs_have_fresh_state(self):
        first = await self.ns["main_graph"].ainvoke({"question": "Laptop specifications"})
        second = await self.ns["main_graph"].ainvoke({"question": "Mouse specifications"})
        self.assertEqual(len(first["results"]), 1)
        self.assertEqual(len(second["results"]), 1)
        self.assertEqual(second["results"][0]["retrieval_attempts"], 1)
        self.assertNotIn("Laptop", self.ns["final_answer_chain"].calls[1]["context"])
        self.assertEqual(self.ns["rewrite_chain"].calls, [])

    async def test_walkthrough_cells_run_in_order_with_offline_dependencies(self):
        ns = bootstrap()
        with contextlib.redirect_stdout(io.StringIO()):
            for index, cell in enumerate(CELLS, 1):
                tags = set(cell["metadata"].get("tags", []))
                if not tags & {"imports", "core", "prompts", "demo"}:
                    continue
                source = "".join(cell["source"])
                compiled = compile(source, f"cell {index}", "exec", ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                result = eval(compiled, ns)
                if inspect.isawaitable(result):
                    await result
                # Install only the IO fakes that have corresponding prompts so far.
                available = {name: value for name, value in self.ns.items() if name.endswith("_chain") or name.endswith("_store") or name == "tavily_search"}
                for name, value in available.items():
                    if name in ns or name.endswith("_store") or name == "tavily_search":
                        ns[name] = value
        self.assertIn("comparison_result", ns)
        self.assertEqual(ns["parallel_result"]["answer"], "Final answer based on retrieved evidence")


if __name__ == "__main__":
    unittest.main()
