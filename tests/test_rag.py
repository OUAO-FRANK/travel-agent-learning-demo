import unittest

from app.rag import KeywordRetriever, RagResult, format_retrieval_context


class KeywordRetrieverTests(unittest.TestCase):
    def test_splits_document_and_returns_ranked_source_chunks(self):
        retriever = KeywordRetriever.from_text(
            "上海历史景点：外滩适合夜景。\n\n上海历史景点：豫园适合白天参观。\n\n杭州西湖适合散步。",
            source="travel-notes.md",
        )

        results = retriever.search("上海 历史景点", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].source, "travel-notes.md")
        self.assertIn("上海历史景点", results[0].text)
        self.assertGreaterEqual(results[0].score, results[1].score)
        self.assertNotIn("杭州西湖", results[0].text + results[1].text)

    def test_returns_no_results_when_query_has_no_matching_terms(self):
        retriever = KeywordRetriever.from_text(
            "上海适合三天旅行。\n\n预算建议为 2000 元。",
            source="travel-notes.md",
        )

        self.assertEqual(retriever.search("成都美食", top_k=3), ())

    def test_rejects_invalid_top_k(self):
        retriever = KeywordRetriever.from_text("上海旅行", source="notes.md")

        with self.assertRaisesRegex(ValueError, "top_k"):
            retriever.search("上海", top_k=0)


class RetrievalFormattingTests(unittest.TestCase):
    def test_formats_results_with_source_and_chunk_id(self):
        results = (
            RagResult(
                chunk_id="travel-notes.md:0",
                source="travel-notes.md",
                text="上海外滩适合夜景。",
                score=2,
            ),
        )

        formatted = format_retrieval_context(results)

        self.assertIn("[travel-notes.md:0]", formatted)
        self.assertIn("上海外滩适合夜景。", formatted)


if __name__ == "__main__":
    unittest.main()
