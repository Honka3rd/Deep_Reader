"""RETRIEVAL-01: actual FAISS identity, publication and read-only recovery gates."""

import json
import os
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
from llama_index.core.schema import TextNode
from retrieval.parsed_document import ParsedDocument
from retrieval.node_record import NodeRecord
from config.faiss_storage_config import FaissStorageConfig
import test_preparation_runtime_safety as safety
from retrieval.index_publication import FILES, JOURNAL, at_directory

ALPHA = "Alpha source describes apples."
BETA = "Beta source describes bananas."
TEXT = ALPHA + "\n" + BETA
rows = []


class Integrity(unittest.TestCase):
    fixture_class = safety.RuntimeSafetyTests

    def setUp(self):
        self.f = self.fixture_class()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.f.path.write_text(TEXT)

        def parse(text, config):
            return ParsedDocument(
                [
                    TextNode(
                        id_="alpha",
                        text=ALPHA,
                        metadata={
                            "chunk_index": 0,
                            "char_start": 0,
                            "char_end": len(ALPHA),
                            "prev_node_id": None,
                            "next_node_id": "beta",
                        },
                    ),
                    TextNode(
                        id_="beta",
                        text=BETA,
                        metadata={
                            "chunk_index": 1,
                            "char_start": len(ALPHA) + 1,
                            "char_end": len(TEXT),
                            "prev_node_id": "alpha",
                            "next_node_id": None,
                        },
                    ),
                ],
                "en",
            )

        self.f.pipeline.node_provider.parse.side_effect = parse
        self.embedder = self.f.pipeline.faiss_index_store.embedder
        self.embedder.get_text_embedding_batch.side_effect = lambda texts: [
            [0.0, 0.0] if t == ALPHA else [10.0, 0.0] for t in texts
        ]
        self.embedder.get_text_embedding.return_value = [0.0, 0.0]
        self.assertEqual(self.f.seed(), TEXT)
        self.config = FaissStorageConfig("book")
        self.store = self.f.pipeline.faiss_index_store
        self.fp = self.f.pipeline.fingerprint_handler
        self.parse_calls = self.f.pipeline.node_provider.parse.call_count

    def files(self):
        return {name: (self.config.base_dir / name).read_bytes() for name in FILES}

    def corrupt(self, kind):
        path = Path(self.config.get_raw_records_path())
        data = json.loads(path.read_text())
        records = data["records"]
        a, b = records["1"], records["2"]
        if kind == "missing_ids":
            data["records"] = {"101": a, "102": b}
        elif kind == "swapped_rows":
            data["records"] = {"1": b, "2": a}
        else:
            raise AssertionError(kind)
        path.write_text(json.dumps(data))

    def inspect(self, kind):
        self.corrupt(kind)
        before = self.files()
        self.assertTrue(self.fp.matches(TEXT, self.config.get_raw_meta_path()))
        result = self.f.pipeline.prepare_and_load("book")
        self.assertEqual(self.files(), before)
        bundle = result.bundle
        hits = [] if bundle is None else bundle.search(ALPHA, top_k=1)
        observation = dict(
            case=kind,
            bundle_ready=result.assets.bundle_ready,
            bundle_present=bundle is not None,
            source_fingerprint_matches=True,
            faiss_ids=(
                []
                if bundle is None
                else faiss.vector_to_array(bundle.faiss_index.id_map).tolist()
            ),
            map_ids=[] if bundle is None else list(bundle.id_to_record),
            record_node_ids=(
                []
                if bundle is None
                else [x.node_id() for x in bundle.id_to_record.values()]
            ),
            hit_texts=[x.text for x in hits],
            parse_calls_before=self.parse_calls,
            parse_calls_after=self.f.pipeline.node_provider.parse.call_count,
            files_unchanged=True,
        )
        rows.append(observation)
        return result

    def test_missing_physical_ids_are_not_accepted_as_ready(self):
        result = self.inspect("missing_ids")
        self.assertFalse(
            result.assets.bundle_ready,
            "FAISS ids absent from the record map must fail validated runtime admission",
        )

    def test_swapped_rows_are_not_accepted_as_ready(self):
        result = self.inspect("swapped_rows")
        self.assertFalse(
            result.assets.bundle_ready,
            "Record node_id/map-key disagreement must not redirect an exact vector match to another source",
        )

    def test_consistent_roundtrip_search_and_readonly_warm_reuse(self):
        before = self.files()
        result = self.f.pipeline.prepare_and_load("book")
        self.assertTrue(result.assets.bundle_ready)
        self.assertEqual(
            [x.text for x in result.bundle.search(ALPHA, top_k=1)], [ALPHA]
        )
        self.assertEqual(before, self.files())
        self.assertEqual(
            self.parse_calls, self.f.pipeline.node_provider.parse.call_count
        )
        rows.append(
            dict(case="valid_roundtrip", ready=True, hit=ALPHA, files_unchanged=True)
        )

    def test_existing_coordinate_guard_still_rejects_invalid_slice(self):
        path = Path(self.config.get_raw_records_path())
        data = json.loads(path.read_text())
        data["records"]["1"]["char_start"] = 1
        path.write_text(json.dumps(data))
        before = self.files()
        result = self.f.pipeline.prepare_and_load("book")
        self.assertFalse(result.assets.bundle_ready)
        self.assertIsNone(result.bundle)
        self.assertEqual(self.files(), before)
        rows.append(
            dict(case="bad_coordinate_control", ready=False, files_unchanged=True)
        )

    def test_candidate_validation_does_not_publish_mismatched_rows(self):
        bundle = self.store.load_validated(self.config, TEXT, self.fp)
        bundle.id_to_record = {1: bundle.id_to_record[2], 2: bundle.id_to_record[1]}
        before = self.files()
        error = None
        try:
            self.store.publish(
                bundle,
                self.config,
                text=TEXT,
                fingerprint=self.fp,
                verify_source=lambda: None,
            )
        except (ValueError, RuntimeError) as exc:
            error = exc
        rows.append(
            dict(
                case="candidate_id_mismatch",
                rejected=error is not None,
                old_trio_preserved=self.files() == before,
                journal_exists=(self.config.base_dir / JOURNAL).exists(),
            )
        )
        with self.subTest(field="admission"):
            self.assertIsNotNone(
                error,
                "The same validated-load gate must reject an inconsistent staged candidate",
            )
        with self.subTest(field="old_trio"):
            self.assertEqual(self.files(), before)

    def payload(self):
        return json.loads(Path(self.config.get_raw_records_path()).read_text())

    def write_payload(self, data):
        Path(self.config.get_raw_records_path()).write_text(json.dumps(data))

    def tree(self):
        return {
            str(p.relative_to(self.config.base_dir)): p.read_bytes()
            for p in self.config.base_dir.rglob("*")
            if p.is_file()
        }

    def assert_runtime_rejected(self):
        before = self.tree()
        with patch.object(
            self.f.pipeline.node_provider,
            "parse",
            side_effect=AssertionError("runtime parse"),
        ), patch.object(
            self.embedder,
            "get_text_embedding_batch",
            side_effect=AssertionError("runtime embeddings"),
        ):
            result = self.f.pipeline.prepare_and_load("book")
        self.assertFalse(result.assets.bundle_ready)
        self.assertIsNone(result.bundle)
        self.assertIn("explicit_prepare_required", str(result.assets.errors))
        self.assertEqual(before, self.tree())

    def test_duplicate_and_noninteger_record_identity_rejected(self):
        baseline = self.payload()
        for value in (2, True, 1.0, "1", None):
            with self.subTest(value=value):
                data = deepcopy(baseline)
                data["records"]["1"]["node_id"] = value
                self.write_payload(data)
                self.assert_runtime_rejected()

    def test_duplicate_json_and_alias_keys_rejected_before_map_collapse(self):
        baseline = self.payload()
        path = Path(self.config.get_raw_records_path())
        for key in ("1", "01", "+1"):
            with self.subTest(key=key):
                # Duplicate JSON keys must not vanish before identity validation.
                first = json.dumps(baseline["records"]["1"])
                second = json.dumps(baseline["records"]["2"])
                prefix = {k: v for k, v in baseline.items() if k != "records"}
                path.write_text(
                    json.dumps(prefix)[:-1]
                    + ',"records":{"1":'
                    + first
                    + ',"'
                    + key
                    + '":'
                    + first
                    + ',"2":'
                    + second
                    + "}}"
                )
                self.assert_runtime_rejected()

    def test_duplicate_or_unmapped_physical_ids_rejected(self):
        for ids in ([1, 1], [101, 102]):
            with self.subTest(ids=ids):
                index = faiss.IndexIDMap(faiss.IndexFlatL2(2))
                index.add_with_ids(
                    np.array([[0.0, 0.0], [10.0, 0.0]], dtype="float32"),
                    np.array(ids, dtype="int64"),
                )
                faiss.write_index(index, self.config.get_raw_index_path())
                self.assert_runtime_rejected()

    def test_noncontiguous_ids_roundtrip_and_publish(self):
        bundle = self.store.load_validated(self.config, TEXT, self.fp)
        index = faiss.IndexIDMap(faiss.IndexFlatL2(2))
        index.add_with_ids(
            np.array([[0.0, 0.0], [10.0, 0.0]], dtype="float32"),
            np.array([7, 101], dtype="int64"),
        )
        data = self.payload()["records"]
        data["1"]["node_id"] = 7
        data["2"]["node_id"] = 101
        bundle.faiss_index = index
        bundle.id_to_record = {
            7: NodeRecord.from_persisted_dict(data["1"]),
            101: NodeRecord.from_persisted_dict(data["2"]),
        }
        self.store.publish(
            bundle,
            self.config,
            text=TEXT,
            fingerprint=self.fp,
            verify_source=lambda: None,
        )
        before = self.tree()
        result = self.f.pipeline.prepare_and_load("book")
        self.assertTrue(result.assets.bundle_ready, result.assets.errors)
        self.assertEqual(set(result.bundle.id_to_record), {7, 101})
        self.assertEqual(
            [x.text for x in result.bundle.search(ALPHA, top_k=1)], [ALPHA]
        )
        self.assertEqual(before, self.tree())

    def test_duplicate_candidate_keeps_active_trio_and_no_journal(self):
        bundle = self.store.load_validated(self.config, TEXT, self.fp)
        bundle.faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(2))
        bundle.faiss_index.add_with_ids(
            np.array([[0.0, 0.0], [10.0, 0.0]], dtype="float32"),
            np.array([1, 1], dtype="int64"),
        )
        before = self.tree()
        with self.assertRaises(ValueError):
            self.store.publish(
                bundle,
                self.config,
                text=TEXT,
                fingerprint=self.fp,
                verify_source=lambda: None,
            )
        self.assertEqual(before, self.tree())
        self.assertFalse((self.config.base_dir / JOURNAL).exists())

    def test_pending_previous_snapshot_uses_same_identity_gate_without_repair(self):
        import retrieval.index_publication as publication

        bundle = self.store.load_validated(self.config, TEXT, self.fp)
        original = publication.os.replace

        def interrupt(source, target):
            original(source, target)
            if str(source).endswith("candidate/index.faiss"):
                raise KeyboardInterrupt()

        with patch.object(publication.os, "replace", side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.store.publish(
                    bundle,
                    self.config,
                    text=TEXT,
                    fingerprint=self.fp,
                    verify_source=lambda: None,
                )
        self.assertTrue((self.config.base_dir / JOURNAL).exists())
        before = self.tree()
        self.assertTrue(self.f.pipeline.prepare_and_load("book").assets.bundle_ready)
        self.assertEqual(before, self.tree())
        previous = publication.previous_config(self.config)
        p = Path(previous.get_raw_records_path())
        data = json.loads(p.read_text())
        data["records"]["1"], data["records"]["2"] = (
            data["records"]["2"],
            data["records"]["1"],
        )
        p.write_text(json.dumps(data))
        self.assert_runtime_rejected()
        self.assertTrue((self.config.base_dir / JOURNAL).exists())

    def test_explicit_prepare_repairs_corrupt_ids_once_then_warm_zero_build(self):
        self.corrupt("swapped_rows")
        self.assert_runtime_rejected()
        builder = self.f.pipeline.faiss_index_builder.build_from_parsed_document
        before = builder.call_count
        prepared = self.f.pipeline.prepare("book")
        self.assertTrue(prepared.bundle_ready, prepared.errors)
        self.assertEqual(builder.call_count, before + 1)
        result = self.f.pipeline.prepare_and_load("book")
        self.assertTrue(result.assets.bundle_ready)
        self.assertEqual(builder.call_count, before + 1)
        self.assertEqual(
            [x.text for x in result.bundle.search(ALPHA, top_k=1)], [ALPHA]
        )


@unittest.skipUnless(
    os.environ.get("DEEP_READER_TEST_POSTGRES_DSN"),
    "requires disposable PostgreSQL DSN",
)
class PostgresIntegrity(Integrity):
    fixture_class = safety.PostgresRuntimeSafetyTests

    @classmethod
    def setUpClass(cls):
        cls.fixture_class.setUpClass()
        cls.addClassCleanup(cls.fixture_class.doClassCleanups)


if __name__ == "__main__":
    unittest.main()
