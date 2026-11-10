from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from dependency_injector import providers
from fastapi import FastAPI
import uvicorn
from learning_routes import router, learning_lifespan
from translation_routes import router as translation_router
from scripts.test_learning_routes import file_container
from scripts.test_learning_file_repository import fixture, configure
from scripts.test_learning_container_wiring import OnceProvider

with TemporaryDirectory(prefix='llrn18-ui-') as root:
 c=file_container(Path(root)); provider=OnceProvider()
 c.llm_provider.override(providers.Object(provider))
 c.document_profile_store.override(providers.Object(SimpleNamespace(load_source=lambda config:SimpleNamespace(document_language_code='en'), load=lambda config:SimpleNamespace(document_language='en'))))
 c.structured_document_artifact_repository().save_reparsed_document(fixture(),'book')
 configure(c.reading_preferences_repository())
 app=FastAPI(lifespan=learning_lifespan);app.state.learning_container=c;app.include_router(router);app.include_router(translation_router);app.state.translation_action=c.translation_action()
 @app.get('/fixture')
 def source():
  return {'sources':[{'id':b.source.content_block_id,'text':b.effective_text} for b in c.learning_repository().read_sources('book').blocks], 'provider_calls':len(provider.calls)}
 uvicorn.run(app,host='127.0.0.1',port=18718,log_level='warning')
 print('PASS temporary file store disposed; provider calls:',len(provider.calls),flush=True)
