"""Exercise the production ASGI app with incremental bodies and no length header."""
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import main


def multipart(payload=b'abc', fields=()):
    body=b''
    for name,value in fields:
        body+=b'--bound\r\nContent-Disposition: form-data; name="'+name.encode()+b'"\r\n\r\n'+value+b'\r\n'
    return body+b'--bound\r\nContent-Disposition: form-data; name="file"; filename="book.txt"\r\nContent-Type: text/plain\r\n\r\n'+payload+b'\r\n--bound--\r\n'


async def streamed(body, *, chunk_size=1024, content_length=None, disconnect=False):
    chunks=[body[i:i+chunk_size] for i in range(0,len(body),chunk_size)]
    calls=0; sent=[]; finished=asyncio.Event()
    async def receive():
        nonlocal calls
        if calls<len(chunks):
            index=calls; calls+=1
            return {'type':'http.request','body':chunks[index], 'more_body':index<len(chunks)-1 or disconnect}
        if disconnect: return {'type':'http.disconnect'}
        await finished.wait()
        return {'type':'http.disconnect'}
    async def send(message):
        sent.append(message)
        if message['type']=='http.response.body' and not message.get('more_body',False): finished.set()
    headers=[(b'content-type',b'multipart/form-data; boundary=bound')]
    if content_length is not None: headers.append((b'content-length',str(content_length).encode()))
    await main.app({'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'POST',
        'scheme':'http','path':'/documents/upload','raw_path':b'/documents/upload','query_string':b'',
        'headers':headers,'client':('127.0.0.1',1),'server':('test',80),'root_path':''},receive,send)
    status=next(m['status'] for m in sent if m['type']=='http.response.start')
    data=json.loads(b''.join(m.get('body',b'') for m in sent if m['type']=='http.response.body'))
    return status,data,calls,len(chunks)


class UploadStreamTests(unittest.TestCase):
    def setUp(self):
        temp=TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root=Path(temp.name)
        p=patch.object(main,'_RAW_DOCUMENT_BASE_DIR',self.root);p.start();self.addCleanup(p.stop)
    def test_file_limit_stops_receiving_without_length_and_cleans_up(self):
        with patch.object(main,'_DOCUMENT_UPLOAD_MAX_BYTES',2048):
            status,data,calls,total=asyncio.run(streamed(multipart(b'x'*12000)))
        self.assertEqual(status,413);self.assertEqual(data['reason'],'file_too_large')
        self.assertLess(calls,total);self.assertEqual(list(self.root.iterdir()),[])
    def test_non_file_field_limit_stops_receiving(self):
        status,data,calls,total=asyncio.run(streamed(multipart(fields=(('title',b'x'*50000),))))
        self.assertEqual(status,413);self.assertEqual(data['reason'],'file_too_large')
        self.assertLess(calls,total);self.assertEqual(list(self.root.iterdir()),[])
    def test_short_length_does_not_bypass_limit(self):
        with patch.object(main,'_DOCUMENT_UPLOAD_MAX_BYTES',2048):
            status,data,calls,total=asyncio.run(streamed(multipart(b'x'*12000),content_length=1))
        self.assertEqual(status,413);self.assertLess(calls,total)
    def test_valid_bytes_and_metadata_survive_single_byte_chunks(self):
        body=multipart(b'Hello\x00world',(('title','中文書'.encode()),('conflict_policy',b'reject')))
        status,data,_,_=asyncio.run(streamed(body,chunk_size=1))
        self.assertEqual(status,201);self.assertEqual(data['title'],'中文書')
        self.assertEqual((self.root/'book.txt').read_bytes(),b'Hello\x00world')
    def test_incomplete_multipart_never_publishes_file(self):
        status,data,_,_=asyncio.run(streamed(multipart(b'abc')[:-14]))
        self.assertEqual(status,400);self.assertEqual(data['reason'],'validation_failure')
        self.assertEqual(list(self.root.iterdir()),[])

    def test_total_request_limit_counts_non_file_overhead(self):
        with patch.object(main,'_DOCUMENT_UPLOAD_REQUEST_MAX_BYTES',4096):
            status,data,calls,total=asyncio.run(streamed(multipart(b'x'*15000)))
        self.assertEqual(status,413);self.assertLess(calls,total)
        self.assertEqual(list(self.root.iterdir()),[])
    def test_oversize_length_rejects_before_receiving(self):
        status,data,calls,total=asyncio.run(streamed(multipart(),content_length=100*1024*1024))
        self.assertEqual(status,413);self.assertEqual(calls,0)
    def test_duplicate_fields_are_rejected(self):
        status,data,_,_=asyncio.run(streamed(multipart(fields=(('title',b'a'),('title',b'b')))))
        self.assertEqual(status,400);self.assertEqual(data['reason'],'validation_failure')
    def test_temp_stream_is_closed_on_success_limit_disconnect_and_parse_failure(self):
        from main_upload import TemporaryFile as original
        streams=[]
        def track(*args,**kwargs):
            stream=original(*args,**kwargs);streams.append(stream);return stream
        with patch('main_upload.TemporaryFile',track):
            for body,kw in ((multipart(),{}),(multipart(fields=(('title',b'x'*50000),)),{}),
                            (multipart()[:-14],{}),(multipart()[:-14],{'disconnect':True})):
                asyncio.run(streamed(body,**kw))
                self.assertTrue(all(stream.closed for stream in streams))
        self.assertEqual(len(streams),4)
        self.assertEqual([p.name for p in self.root.iterdir()],['book.txt'])
    def test_real_backup_excerpt_upload_hash_and_bytes(self):
        # Private corpus is optional locally; deterministic transport cases run everywhere.
        import hashlib
        source=Path('data/test-books/fa794acef055f235')
        candidates=list(source.glob('**/*.txt')) if source.exists() else []
        if not candidates: self.skipTest('private backed-up TXT corpus unavailable')
        payload=candidates[0].read_bytes()[:65536]
        self.assertGreater(len(payload),0)
        status,data,_,_=asyncio.run(streamed(multipart(payload),chunk_size=777))
        self.assertEqual(status,201);self.assertEqual(data['source_hash'],hashlib.sha256(payload).hexdigest())
        self.assertEqual((self.root/'book.txt').read_bytes(),payload)

if __name__=='__main__': unittest.main()
