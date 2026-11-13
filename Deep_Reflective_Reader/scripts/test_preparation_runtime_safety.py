"""Preparation audit regressions with real stores/FAISS and deterministic providers."""
import os
import tempfile
import unittest
import json
from pathlib import Path
from dataclasses import replace
from unittest.mock import Mock, patch
from uuid import uuid4

from test_prepare_structured_reuse_before_raw_load import _build_pipeline, _StructuredStore, _fake_document
from document_structure.structured_document_builder import StructuredDocumentBuilder
from document_structure.structured_document_store import StructuredDocumentStore
from document_structure.structured_document_artifact_repository import StructuredDocumentArtifactRepository
from doc_loaders.text_document_loader import TextDocumentLoader
from doc_loaders.pdf_document_loader import PdfDocumentLoader
from profile.document_profile_store import DocumentProfileStore
from config.faiss_storage_config import FaissStorageConfig
from fingerprint_handler import FingerprintHandler
from retrieval.faiss_index_store import FaissIndexStore
from test_pdf_page_inspection import page_fixture
from bundle_provider import BundleProvider
from bundle_factory import BundleFactory
from retrieval.faiss_index_builder import FaissIndexBuilder
from retrieval.parsed_document import ParsedDocument
from llama_index.core.schema import TextNode
from section_tasks.note_interaction_service import NoteInteractionService
from language.language_code import LanguageCode
from retrieval.index_publication import FILES, JOURNAL

ROOT = Path(__file__).resolve().parents[1]

def setup_pipeline(path):
    store = _StructuredStore(exists_value=False, document=_fake_document("book"))
    loader = TextDocumentLoader(path.parent)
    pipeline = _build_pipeline(loader=loader, store=store)
    pipeline.structured_document_builder = StructuredDocumentBuilder()
    pipeline.profile_store = DocumentProfileStore()
    embedder = Mock()
    embedder.probe_vector_dimension.return_value = 2
    embedder.get_text_embedding_batch.side_effect = lambda texts: [[1.0, float(len(t))] for t in texts]
    llm = Mock()
    llm.get_model_capabilities.return_value = None
    budgets = dict(target_max_input_tokens=4096, target_max_output_tokens=512,
        target_max_context_tokens=2048, input_budget_utilization_ratio=.8,
        context_budget_utilization_ratio=.5)
    pipeline.faiss_index_store = FaissIndexStore(embedder, llm, **budgets)
    pipeline.fingerprint_handler = FingerprintHandler(
        embedding_model="deterministic-audit", chunk_size=512, chunk_overlap=0)
    pipeline.node_provider = Mock()
    pipeline.node_provider.parse.side_effect = lambda text, config: ParsedDocument(
        [TextNode(text=text, metadata={"chunk_index": 0, "char_start": 0,
            "char_end": len(text), "prev_node_id": None, "next_node_id": None})],
        document_language="en")
    pipeline.faiss_index_builder = Mock(wraps=FaissIndexBuilder(embedder, llm, batch_size=10, **budgets))
    factory = BundleFactory(builder=pipeline.faiss_index_builder,
        fingerprint_handler=pipeline.fingerprint_handler, store=pipeline.faiss_index_store,
        node_provider=pipeline.node_provider, profile_builder=pipeline.profile_builder,
        profile_store=pipeline.profile_store)
    provider = BundleProvider(storage_config_factory=FaissStorageConfig,
        fingerprint_handler_factory=lambda: pipeline.fingerprint_handler,
        bundle_factory_provider=lambda **kw: factory, loader_factory=pipeline.loader_factory)
    pipeline.bundle_provider = Mock(wraps=provider)
    return pipeline, store, loader


class RuntimeSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        previous = os.getcwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, previous)
        self.path = Path(self.temp.name) / "book.txt"
        self.path.write_text("Original accepted body with a full paragraph.\n")
        self.pipeline, _, self.loader = setup_pipeline(self.path)
        self.store = StructuredDocumentStore()
        self.pipeline.structured_document_store = self.store

    def config(self):
        return self.pipeline._structured_config("book")

    def seed(self, kind="txt", mode="fast", *, index=True):
        if kind == "pdf":
            self.path.unlink(missing_ok=True)
            self.path = Path(self.temp.name) / "book.pdf"
            writer, _ = page_fixture(
                b"BT /F1 12 Tf 10 70 Td (A readable paragraph about science and evidence.) Tj ET",
                with_font=True)
            writer.write(self.path)
            self.loader = PdfDocumentLoader(self.path.parent, ocr_enabled=False)
            self.pipeline.loader_factory.loader = self.loader
        artifact = self.loader.open_source("book", mode="fast").extract()
        # Accepted older extraction modes are persisted evidence, not a request
        # to run an OCR engine in these runtime/read-only tests.
        result = replace(artifact.result, identity=replace(artifact.result.identity,
            mode=mode, extraction_version="accepted-older-version"))
        document = StructuredDocumentBuilder().build(document_id="book", title="book",
            raw_text=result.canonical_text, language=LanguageCode.EN)
        document = replace(document, source_path=str(self.path), parse_provenance={
            **document.parse_provenance, "extraction": result.record(),
            "source_page_spans": [dict(page_index=b.page_index, char_start=b.char_start,
                char_end=b.char_end, page_label=b.page_label) for b in artifact.page_boundaries]})
        self.store.save(document, self.config())
        profile = self.pipeline.profile_builder.build(text=result.canonical_text, document_language="en")
        self.pipeline.profile_store.save(profile, FaissStorageConfig("book"))
        if index:
            assets = self.pipeline.prepare("book")
            self.assertTrue(assets.bundle_ready, assets.errors)
        return result.canonical_text

    def files(self):
        base = FaissStorageConfig("book").base_dir
        return {name: (base / name).read_bytes() for name in FILES if (base / name).exists()}

    def repository(self):
        return StructuredDocumentArtifactRepository(store=self.store)

    def test_initial_runtime_missing_requires_explicit_prepare(self):
        with patch.object(self.loader, "extract", side_effect=AssertionError("implicit initial extraction")):
            missing = self.pipeline.prepare_and_load("book")
        self.assertIsNone(missing.bundle)
        self.assertFalse(missing.assets.structured_document_ready)
        prepared = self.pipeline.prepare("book")
        self.assertTrue(prepared.bundle_ready, prepared.errors)
        self.assertIsNotNone(self.pipeline.prepare_and_load("book").bundle)

    def test_accepted_modes_preserve_notes_ids_versions_and_warm_work(self):
        for kind, mode in [("txt", "fast"), ("pdf", "fast"), ("pdf", "auto"), ("pdf", "ocr")]:
            with self.subTest(kind=kind, mode=mode):
                self.seed(kind, mode)
                notes = NoteInteractionService(self.repository())
                state = notes.read("book", target_level="document")
                revision = state.get("revision", 0)
                if state.get("note"):
                    revision = state["note"]["revision"]
                notes.save("book", target_level="document", text="Keep this note",
                    expected_revision=revision, mutation_id=uuid4().hex, target_token=state["target_token"])
                from shared.task_artifacts import SummaryArtifact, TaskArtifacts
                document = self.store.load(self.config())
                # Store a second interaction through the hierarchy owner, then
                # compare the entire readback (not only note text or counts).
                chapter = document.chapters[0]
                document = replace(document, chapters=[replace(chapter,
                    task_artifacts=TaskArtifacts(summary=SummaryArtifact(content="Accepted summary"))),
                    *document.chapters[1:]])
                self.repository().save_document(document, "book")
                before = self.store.load(self.config()).to_dict()
                old = self.files()
                with patch.object(self.loader, "open_source", side_effect=AssertionError("capture/extract")), \
                     patch.object(self.pipeline.language_detector, "detect", side_effect=AssertionError("language")), \
                     patch.object(self.pipeline.profile_builder, "build", side_effect=AssertionError("profile")), \
                     patch.object(self.pipeline.faiss_index_builder, "build_from_parsed_document", side_effect=AssertionError("embedding")):
                    from app.qa_coordinator import QACoordinator
                    coordinator = object.__new__(QACoordinator)
                    coordinator.document_preparation_pipeline = self.pipeline
                    self.assertIsNotNone(coordinator.get_bundle("book"))
                self.assertEqual(before, self.store.load(self.config()).to_dict())
                self.assertEqual(old, self.files())
                self.assertIsNotNone(notes.read("book", target_level="document")["note"])

    def test_runtime_missing_or_changed_assets_never_rebuild(self):
        self.seed()
        before = self.store.load(self.config()).to_dict()
        old = self.files()
        self.path.write_text("Changed source.\n")
        with patch.object(self.loader, "extract", side_effect=AssertionError("extract")), \
             patch.object(self.pipeline.faiss_index_builder, "build_from_parsed_document", side_effect=AssertionError("build")):
            result = self.pipeline.prepare_and_load("book")
        self.assertIsNone(result.bundle)
        self.assertFalse(result.assets.bundle_ready)
        self.assertEqual(old, self.files())
        self.assertEqual(before, self.store.load(self.config()).to_dict())
        self.path.write_text("Original accepted body with a full paragraph.\n")
        for name in FILES:
            path = FaissStorageConfig("book").base_dir / name
            saved = path.read_bytes()
            path.unlink()
            with patch.object(self.pipeline.faiss_index_builder, "build_from_parsed_document", side_effect=AssertionError("build")):
                result = self.pipeline.prepare_and_load("book")
            self.assertIsNone(result.bundle)
            self.assertFalse(path.exists())
            path.write_bytes(saved)

    def test_manual_and_page_reads_use_exact_accepted_snapshot(self):
        for kind in ("txt", "pdf"):
            with self.subTest(kind=kind):
                text = self.seed(kind)
                document = self.store.load(self.config())
                before = document.to_dict()
                with patch.object(self.loader, "extract", side_effect=AssertionError("extract")), \
                     patch.object(self.loader, "load", side_effect=AssertionError("load")), \
                     patch.object(self.pipeline.language_detector, "detect", side_effect=AssertionError("language")):
                    evidence = self.pipeline.load_manual_structure_source_evidence("book")
                    with patch.object(self.store, "load", side_effect=AssertionError("snapshot reload")):
                        pages = self.pipeline.load_task_layout_page_boundaries("book", document=document)
                self.assertEqual(text, evidence.raw_text)
                self.assertEqual(len(pages), 1 if kind == "pdf" else 0)
                if pages:
                    self.assertEqual(pages[0].text, text)
                self.assertEqual(before, self.store.load(self.config()).to_dict())

    def test_invalid_canonical_evidence_and_missing_optional_pages(self):
        self.seed("pdf")
        document = self.store.load(self.config())
        section = document.chapters[0].sections[0]
        cases = [replace(document, raw_text="", chapters=[replace(document.chapters[0], sections=[
            replace(section, char_start=start, char_end=end, content=content)])])
            for start, end, content in [(1,section.char_end,section.content),
                (-1,section.char_end,section.content), (0,section.char_end-1,section.content[:-1])]]
        cases += [replace(case, raw_text=self.pipeline.load_manual_structure_source_evidence("book").raw_text)
                  for case in list(cases)]
        overlapping = [section, replace(section, section_id="overlap")]
        cases.append(replace(document, raw_text="", chapters=[replace(document.chapters[0], sections=overlapping)]))
        cases.append(replace(document, parse_provenance={}))
        cases.append(replace(document, parse_provenance={**document.parse_provenance,
            "extraction": {**document.parse_provenance["extraction"], "identity": {
                **document.parse_provenance["extraction"]["identity"], "source_sha256": "0"*64}}}))
        for key, value in [("text_sha256", "0"*64), ("text_length", 999), ("completion", "partial")]:
            cases.append(replace(document, parse_provenance={**document.parse_provenance,
                "extraction": {**document.parse_provenance["extraction"], key:value}}))
        for invalid in cases:
            with self.subTest(invalid=invalid.parse_provenance):
                with patch.object(self.store, "load", return_value=invalid), \
                     patch.object(self.loader, "load", side_effect=AssertionError("invalid-source extraction")):
                    with self.assertRaises(ValueError):
                        self.pipeline.load_manual_structure_source_evidence("book")
                    self.assertEqual([], self.pipeline.load_task_layout_page_boundaries("book", document=invalid))
        no_pages = replace(document, parse_provenance={k:v for k,v in document.parse_provenance.items() if k != "source_page_spans"})
        with patch.object(self.store, "load", return_value=no_pages):
            self.assertEqual([], self.pipeline.load_manual_structure_source_evidence("book").page_boundaries)

    def test_source_changes_at_each_expensive_stage_preserve_old_index(self):
        for stage in ("extract", "structure", "embedding", "save"):
            with self.subTest(stage=stage):
                self.seed()
                old = self.files()
                old_document = self.store.load(self.config()).to_dict()
                target, method = {"extract": (self.loader, "extract"),
                    "structure": (self.pipeline.structured_document_builder, "build"),
                    "embedding": (self.pipeline.faiss_index_builder, "build_from_parsed_document"),
                    "save": (self.pipeline.faiss_index_store, "save")}[stage]
                original = getattr(target, method)
                def change(*args, **kwargs):
                    result = original(*args, **kwargs)
                    self.path.write_text("Changed during " + stage)
                    return result
                with patch.object(target, method, side_effect=change), \
                     patch.object(self.pipeline.faiss_index_store, "clear", side_effect=AssertionError("clear")):
                    result = self.pipeline.prepare_and_load("book", force_rebuild=True)
                self.assertIsNone(result.bundle)
                self.assertFalse(result.assets.faiss_ready)
                self.assertEqual(old, self.files())
                if stage in ("extract", "structure"):
                    self.assertEqual(old_document, self.store.load(self.config()).to_dict())
                self.pipeline.faiss_index_store.load(FaissStorageConfig("book"))
                self.path.write_text("Original accepted body with a full paragraph.\n")

    def test_build_and_save_failures_do_not_clear_or_retry(self):
        for existing in (False, True):
            for stage in ("parse", "embedding", "save", "fingerprint"):
                with self.subTest(existing=existing, stage=stage):
                    self.seed(index=existing)
                    if not existing:
                        for name in FILES:
                            (FaissStorageConfig("book").base_dir/name).unlink(missing_ok=True)
                    old = self.files()
                    target, method = {"parse": (self.pipeline.node_provider,"parse"),
                        "embedding": (self.pipeline.faiss_index_builder,"build_from_parsed_document"),
                        "save": (self.pipeline.faiss_index_store,"save"),
                        "fingerprint": (self.pipeline.fingerprint_handler,"save")}[stage]
                    with patch.object(target, method, side_effect=RuntimeError("injected")) as failed, \
                         patch.object(self.pipeline.bundle_provider,"load_prepared_bundle",side_effect=AssertionError("retry")):
                        assets = self.pipeline.prepare("book", force_rebuild=True)
                    self.assertFalse(assets.faiss_ready)
                    self.assertFalse(assets.bundle_ready)
                    self.assertEqual(failed.call_count,1)
                    self.assertEqual(old,self.files())
                    if existing:
                        self.pipeline.faiss_index_store.load(FaissStorageConfig("book"))

    def test_publish_interruptions_leave_recoverable_previous_trio(self):
        import retrieval.index_publication as publication
        class Interrupted(BaseException):
            pass
        for existing in (False, True):
            for position in range(3):
                with self.subTest(existing=existing, position=position):
                    text = self.seed(index=existing)
                    config = FaissStorageConfig("book")
                    if not existing:
                        for name in FILES:
                            (config.base_dir/name).unlink(missing_ok=True)
                    before = self.files()
                    bundle = self.pipeline.faiss_index_builder.build_from_parsed_document(
                        self.pipeline.node_provider.parse(text,config))
                    original = publication.os.replace
                    count = 0
                    def interrupt(source, destination):
                        nonlocal count
                        if Path(source).parent.name == "candidate":
                            count += 1
                            if count == position + 1:
                                raise Interrupted()
                        return original(source,destination)
                    with patch.object(publication.os,"replace",side_effect=interrupt):
                        with self.assertRaises(Interrupted):
                            self.pipeline.faiss_index_store.publish(bundle,config,text=text,
                                fingerprint=self.pipeline.fingerprint_handler,verify_source=lambda:None)
                    self.assertTrue((config.base_dir/JOURNAL).exists())
                    interrupted_files = {str(p):p.read_bytes() for p in config.base_dir.rglob("*") if p.is_file()}
                    if existing:
                        self.pipeline.faiss_index_store.load_validated(config,text,self.pipeline.fingerprint_handler)
                    else:
                        with self.assertRaises(ValueError):
                            self.pipeline.faiss_index_store.load_validated(config,text,self.pipeline.fingerprint_handler)
                    self.assertEqual(interrupted_files,
                        {str(p):p.read_bytes() for p in config.base_dir.rglob("*") if p.is_file()})
                    self.pipeline.faiss_index_store.recover(config)

                    self.assertEqual(before,self.files())
                    self.assertFalse((config.base_dir/JOURNAL).exists())
                    self.pipeline.faiss_index_store.recover(config)

    def test_switch_exceptions_and_post_switch_source_failure_roll_back(self):
        import retrieval.index_publication as publication
        for position in range(4):
            with self.subTest(position=position):
                text = self.seed()
                config = FaissStorageConfig("book")
                before = self.files()
                bundle = self.pipeline.faiss_index_builder.build_from_parsed_document(
                    self.pipeline.node_provider.parse(text, config))
                original = publication.os.replace
                count = 0
                def fail(source, destination):
                    nonlocal count
                    if Path(source).parent.name == "candidate":
                        count += 1
                        if count == position + 1:
                            raise OSError("switch failure")
                    return original(source, destination)
                verify = Mock(side_effect=[None, ValueError("source changed")] if position == 3 else None)
                with patch.object(publication.os, "replace", side_effect=fail):
                    with self.assertRaises((OSError, ValueError)):
                        self.pipeline.faiss_index_store.publish(bundle, config, text=text,
                            fingerprint=self.pipeline.fingerprint_handler, verify_source=verify)
                self.assertEqual(before, self.files())
                self.pipeline.faiss_index_store.load_validated(config, text, self.pipeline.fingerprint_handler)

    def test_process_exit_during_switch_preserves_readable_previous_snapshot(self):
        import multiprocessing
        import retrieval.index_publication as publication
        text = self.seed()
        config = FaissStorageConfig("book")
        before = self.files()
        bundle = self.pipeline.faiss_index_builder.build_from_parsed_document(
            self.pipeline.node_provider.parse(text, config))
        def terminated_writer():
            original = publication.os.replace
            def replace_then_exit(source, destination):
                original(source, destination)
                if Path(source).parent.name == "candidate":
                    os._exit(73)
            with patch.object(publication.os, "replace", side_effect=replace_then_exit):
                self.pipeline.faiss_index_store.publish(bundle, config, text=text,
                    fingerprint=self.pipeline.fingerprint_handler, verify_source=lambda: None)
        process = multiprocessing.get_context("fork").Process(target=terminated_writer)
        process.start()
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
        self.assertEqual(process.exitcode, 73)
        self.pipeline.faiss_index_store.load_validated(config, text, self.pipeline.fingerprint_handler)
        self.pipeline.faiss_index_store.recover(config)
        self.assertEqual(before, self.files())

    def test_commit_directory_sync_failure_restores_previous_files(self):
        import retrieval.index_publication as publication
        text = self.seed()
        config = FaissStorageConfig("book")
        before = self.files()
        bundle = self.pipeline.faiss_index_builder.build_from_parsed_document(
            self.pipeline.node_provider.parse(text, config))
        original = publication.sync_directory
        observed_journal = False
        failed = False
        def fail_after_unlink(path):
            nonlocal observed_journal, failed
            present = (config.base_dir/JOURNAL).exists()
            if Path(path) == config.base_dir and observed_journal and not present and not failed:
                failed = True
                raise OSError("commit sync failure")
            observed_journal = observed_journal or present
            return original(path)
        with patch.object(publication, "sync_directory", side_effect=fail_after_unlink):
            with self.assertRaisesRegex(OSError, "commit sync"):
                self.pipeline.faiss_index_store.publish(bundle, config, text=text,
                    fingerprint=self.pipeline.fingerprint_handler, verify_source=lambda: None)
        self.assertTrue(failed)
        self.assertEqual(before, self.files())

    def test_partial_binary_save_failure_preserves_old_assets(self):
        import retrieval.faiss_index_store as storage
        for existing in (False, True):
            with self.subTest(existing=existing):
                self.seed(index=existing)
                before = self.files()
                original = storage.faiss.write_index
                def partial(index, path):
                    original(index, path)
                    raise OSError("after binary write, before records")
                with patch.object(storage.faiss, "write_index", side_effect=partial) as save:
                    assets = self.pipeline.prepare("book", force_rebuild=True)
                self.assertFalse(assets.faiss_ready)
                self.assertFalse(assets.bundle_ready)
                self.assertEqual(save.call_count, 1)
                self.assertEqual(before, self.files())
                self.assertFalse(list(FaissStorageConfig("book").base_dir.glob(".index-candidate-*")))


@unittest.skipUnless(os.environ.get("DEEP_READER_TEST_POSTGRES_DSN"), "requires disposable PostgreSQL DSN")
class PostgresRuntimeSafetyTests(RuntimeSafetyTests):
    @classmethod
    def setUpClass(cls):
        import psycopg
        from psycopg import sql
        from psycopg.conninfo import make_conninfo
        cls.dsn = os.environ["DEEP_READER_TEST_POSTGRES_DSN"]
        cls.schema = "prep_safety_" + uuid4().hex
        with psycopg.connect(cls.dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(cls.schema)))
        cls.addClassCleanup(cls.drop_schema)
        cls.scoped = make_conninfo(cls.dsn, options=f"-c search_path={cls.schema}")
        with psycopg.connect(cls.scoped) as connection:
            for path in sorted((ROOT/"db/migrations").glob("*.sql")):
                connection.execute(path.read_text())

    @classmethod
    def drop_schema(cls):
        import psycopg
        from psycopg import sql
        with psycopg.connect(cls.dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(cls.schema)))

    def setUp(self):
        super().setUp()
        from db.postgres_structured_document_store import PostgresStructuredDocumentStore
        self.store = PostgresStructuredDocumentStore(self.scoped, namespace=uuid4().hex)
        self.pipeline.structured_document_store = self.store

    def repository(self):
        from db.postgres_structured_document_artifact_repository import PostgresStructuredDocumentArtifactRepository
        return PostgresStructuredDocumentArtifactRepository(self.store)
