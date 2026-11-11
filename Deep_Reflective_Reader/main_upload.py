"""Bounded multipart transport for the main upload route; no preparation policy."""
from dataclasses import dataclass
from hashlib import sha256
from tempfile import TemporaryFile
from typing import BinaryIO

from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header
from python_multipart.exceptions import MultipartParseError


@dataclass
class ReceivedUpload:
    stream: BinaryIO
    filename: str = ''
    size: int = 0
    title: str | None = None
    conflict_policy: str = 'reject'
    source_hash: str = ''


async def receive_upload(request, *, file_limit, request_limit, field_limit=16*1024):
    content_type = request.headers.get('content-type', '')
    if len(content_type) > 1024:
        raise ValueError('validation_failure')
    media, options = parse_options_header(content_type)
    boundary = options.get(b'boundary', b'')
    if media != b'multipart/form-data' or not 1 <= len(boundary) <= 200:
        raise ValueError('validation_failure')
    length = request.headers.get('content-length')
    if length is not None:
        if not length.isascii() or not length.isdigit() or len(length) > 20:
            raise ValueError('validation_failure')
        if int(length) > request_limit:
            raise ValueError('file_too_large')

    result = ReceivedUpload(TemporaryFile(mode='w+b'))
    digest = sha256()
    fields = {}; headers = {}; header_name = bytearray(); header_value = bytearray()
    field = bytearray(); name = None; header_bytes = 0; parts = 0; ended = False

    def part_begin():
        nonlocal header_bytes, parts
        parts += 1
        if parts > 3:
            raise ValueError('file_too_large')
        headers.clear(); field.clear(); header_bytes = 0

    def header_begin():
        header_name.clear(); header_value.clear()

    def header_data(target, data, start, end):
        nonlocal header_bytes
        header_bytes += end-start
        if header_bytes > 8192:
            raise ValueError('file_too_large')
        target.extend(data[start:end])

    def header_end():
        key = bytes(header_name).lower()
        if key in headers:
            raise ValueError('validation_failure')
        headers[key] = bytes(header_value)

    def headers_finished():
        nonlocal name
        disposition, params = parse_options_header(headers.get(b'content-disposition', b''))
        name = params.get(b'name')
        if disposition != b'form-data' or name not in (b'file', b'title', b'conflict_policy') or name in fields:
            raise ValueError('validation_failure')
        fields[name] = True
        if b'content-transfer-encoding' in headers:
            raise ValueError('validation_failure')
        if name == b'file':
            filename = params.get(b'filename')
            if filename is None:
                raise ValueError('missing_source')
            result.filename = filename.decode('utf-8')
        elif b'filename' in params:
            raise ValueError('validation_failure')

    def part_data(data, start, end):
        size = end-start
        if name == b'file':
            if result.size+size > file_limit:
                raise ValueError('file_too_large')
            result.size += size
            chunk = memoryview(data)[start:end]
            result.stream.write(chunk); digest.update(chunk)
        else:
            if len(field)+size > field_limit:
                raise ValueError('file_too_large')
            field.extend(data[start:end])

    def part_end():
        if name == b'title':
            result.title = field.decode('utf-8', errors='replace').strip() or None
        elif name == b'conflict_policy':
            result.conflict_policy = field.decode('utf-8', errors='replace').strip().lower().replace('-', '_') or 'reject'

    def end():
        nonlocal ended
        ended = True

    parser = MultipartParser(boundary, {
        'on_part_begin':part_begin, 'on_header_begin':header_begin,
        'on_header_field':lambda d,s,e: header_data(header_name,d,s,e),
        'on_header_value':lambda d,s,e: header_data(header_value,d,s,e),
        'on_header_end':header_end, 'on_headers_finished':headers_finished,
        'on_part_data':part_data, 'on_part_end':part_end, 'on_end':end,
    }, max_header_count=8, max_header_size=4096)
    try:
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
            if received > request_limit:
                raise ValueError('file_too_large')
            # Also bound callback writes if a server supplies a large ASGI chunk.
            for start in range(0, len(chunk), 64*1024):
                parser.write(chunk[start:start+64*1024])
        parser.finalize()
        if not ended:
            raise ValueError('validation_failure')
        if b'file' not in fields:
            raise ValueError('missing_source')
        if result.conflict_policy not in {'reject', 'replace', 'create_distinct'}:
            raise ValueError('validation_failure')
        result.source_hash = digest.hexdigest()
        result.stream.seek(0)
        return result
    except MultipartParseError:
        result.stream.close()
        raise ValueError('validation_failure') from None
    except BaseException:
        result.stream.close()
        raise
