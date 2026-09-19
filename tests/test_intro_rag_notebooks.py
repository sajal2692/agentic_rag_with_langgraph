"""Run the basic and router notebook graphs without external API calls."""

import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nbformat
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
BASIC = NOTEBOOKS / "01_basic_rag.ipynb"
ROUTER = NOTEBOOKS / "02_agentic_router_rag.ipynb"


class FakeStore:
    def __init__(self, owner, source):
        self.owner = owner
        self.source = source

    def similarity_search(self, query, k):
        self.owner.calls.append(("retrieve", self.source, query, k))
        if self.owner.empty:
            return []
        return [Document(
            page_content=f"Evidence from {self.source} for {query}",
            metadata={"source": f"{self.source}.csv"},
        )]


class FakeModel:
    def __init__(self, owner, name):
        self.owner = owner
        self.name = name

    def invoke(self, messages):
        text = messages.to_string()
        self.owner.calls.append(("generate", self.name, text))
        return SimpleNamespace(content="Answer " + hashlib.sha256(text.encode()).hexdigest()[:12])

    def with_structured_output(self, schema):
        def route(messages):
            self.owner.calls.append(("route", messages.to_string()))
            return schema(chosen_collection=self.owner.source, reasoning="Fixture route")
        return RunnableLambda(route)


class Fixtures:
    def __init__(self, source="catalog", empty=False):
        self.source = source
        self.empty = empty
        self.calls = []

    def bind(self, namespace):
        for variable, source in [
            ("vector_store", "combined"), ("catalog_store", "catalog"),
            ("faq_store", "faq"), ("troubleshooting_store", "troubleshooting"),
        ]:
            namespace[variable] = FakeStore(self, source)
        for name in ("llm", "generation_llm", "routing_llm"):
            namespace[name] = FakeModel(self, name)
        namespace["tavily_api_key"] = "offline-test-key"
        namespace["TavilySearch"] = lambda **kwargs: SimpleNamespace(invoke=self.search_web)

    def search_web(self, inputs):
        self.calls.append(("web_search", dict(inputs)))
        results = [] if self.empty else [{
            "content": f"Web evidence for {inputs['query']}",
            "url": "https://example.test/support",
            "title": "Support fixture",
        }]
        return {"results": results}


def load_notebook(path, fixtures, include_demos=False):
    """Execute definitions in notebook order, skipping credentials and real clients."""
    namespace = {}
    cells = json.loads(path.read_text())["cells"]
    selected = {"core", "prompts"} | ({"demo"} if include_demos else set())
    with contextlib.redirect_stdout(io.StringIO()):
        for index, cell in enumerate(cells, 1):
            if cell["cell_type"] != "code":
                continue
            tags = set(cell["metadata"].get("tags", []))
            source = "".join(cell["source"])
            filename = f"{path.name}:cell {index}"
            if "imports" in tags:
                # Import the real libraries without loading the user's .env file.
                tree = ast.parse(source)
                tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
                exec(compile(tree, filename, "exec"), namespace)
                fixtures.bind(namespace)
            elif tags & selected:
                exec(compile(source, filename, "exec"), namespace)
    return namespace


class IntroRAGTests(unittest.TestCase):
    def setUp(self):
        tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"})
        tracing.start()
        self.addCleanup(tracing.stop)

    def test_notebook_schema_syntax_and_navigation(self):
        for path in (BASIC, ROUTER):
            with self.subTest(notebook=path.name):
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                for index, cell in enumerate(notebook.cells, 1):
                    if cell.cell_type == "code":
                        compile(cell.source, f"cell {index}", "exec")
                markdown = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
                links = set(re.findall(r"\]\(#([^)]*)\)", markdown))
                anchors = set(re.findall(r'<a id="([^"]+)"', markdown))
                self.assertEqual(len(links), 5)
                self.assertLessEqual(links, anchors)

    def test_basic_graph_preserves_question_and_passes_documents_to_generation(self):
        fixtures = Fixtures()
        ns = load_notebook(BASIC, fixtures)
        with contextlib.redirect_stdout(io.StringIO()):
            result = ns["ask_question"]("Laptop specifications")
        self.assertEqual(result["question"], "Laptop specifications")
        self.assertEqual(fixtures.calls[0], ("retrieve", "combined", "Laptop specifications", 5))
        self.assertEqual(len(fixtures.calls), 2)
        self.assertIn("Source: combined.csv\nContent: Evidence from combined", fixtures.calls[1][2])
        self.assertEqual(result["context"][0].metadata["source"], "combined.csv")
        self.assertTrue(result["answer"].startswith("Answer "))

    def test_router_runs_only_the_selected_source_and_preserves_its_reason(self):
        for source in ("catalog", "faq", "troubleshooting", "web_search"):
            with self.subTest(source=source):
                fixtures = Fixtures(source)
                ns = load_notebook(ROUTER, fixtures)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = ns["ask_router_rag"]("A customer question")
                self.assertEqual(result["query"], "A customer question")
                self.assertEqual(result["chosen_collection"], source)
                self.assertEqual(result["routing_reasoning"], "Fixture route")
                self.assertEqual([call[0] for call in fixtures.calls], [
                    "route", "web_search" if source == "web_search" else "retrieve", "generate",
                ])
                self.assertIn(f"Context from {source}", fixtures.calls[-1][2])
                if source == "web_search":
                    self.assertEqual(result["retrieved_docs"][0].metadata, {
                        "source": "https://example.test/support", "title": "Support fixture",
                    })
                    self.assertEqual(fixtures.calls[1], ("web_search", {"query": "A customer question"}))
                else:
                    self.assertEqual(fixtures.calls[1], ("retrieve", source, "A customer question", 5))

    def test_empty_retrieval_still_reaches_generation_without_an_added_retry(self):
        for path, helper, documents in [
            (BASIC, "ask_question", "context"), (ROUTER, "ask_router_rag", "retrieved_docs"),
        ]:
            with self.subTest(notebook=path.name):
                fixtures = Fixtures(empty=True)
                ns = load_notebook(path, fixtures)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = ns[helper]("Missing information")
                self.assertEqual(result[documents], [])
                self.assertEqual(sum(call[0] == "retrieve" for call in fixtures.calls), 1)
                self.assertEqual(sum(call[0] == "generate" for call in fixtures.calls), 1)

    def test_basic_examples_run_in_order_and_retain_inspectable_results(self):
        ns = load_notebook(BASIC, Fixtures(), include_demos=True)
        expected = ["shipping", "product", "support", "custom", "multi_domain",
                    "ports", "returns", "comparison", "windows", "gpu"]
        for name in expected:
            self.assertIn("context", ns[f"{name}_result"])
            self.assertIn("answer", ns[f"{name}_result"])

    def test_router_examples_run_in_order_and_retain_inspectable_results(self):
        ns = load_notebook(ROUTER, Fixtures(), include_demos=True)
        for name in ("recommendation", "shipping", "support", "ports", "returns", "windows", "gpu"):
            self.assertIn("retrieved_docs", ns[f"{name}_result"])
            self.assertIn("answer", ns[f"{name}_result"])


if __name__ == "__main__":
    unittest.main()
