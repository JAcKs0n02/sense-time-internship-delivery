"""Bounded single-image decoding; no path reads or remote image fetches."""
import base64
import hashlib
import io
import math
import warnings

from PIL import Image, ImageOps

from .service_contract import request_payload

MAX_BYTES = 10 * 1024 * 1024
MAX_SOURCE_PIXELS = 20_000_000
MAX_OUTPUT_PIXELS = 401408


def prepare_image(data):
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_BYTES:
        raise ValueError('请上传不超过10MiB的JPEG、PNG或WEBP图片。')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {'PNG', 'JPEG', 'WEBP'}:
                    raise ValueError('unsupported format')
                width, height = source.size
                if width * height > MAX_SOURCE_PIXELS or min(width, height) < 1:
                    raise ValueError('source dimensions too large')
                if max(width, height) / min(width, height) > 200:
                    raise ValueError('extreme aspect ratio')
                if getattr(source, 'n_frames', 1) != 1:
                    raise ValueError('animated image')
                source.verify()
            with Image.open(io.BytesIO(data)) as source:
                source.load()
                oriented = ImageOps.exif_transpose(source)
                rgba = oriented.convert('RGBA')
                image = Image.new('RGB', rgba.size, 'white')
                image.paste(rgba, mask=rgba.getchannel('A'))
                if image.width * image.height > MAX_OUTPUT_PIXELS:
                    ratio = math.sqrt(MAX_OUTPUT_PIXELS / (image.width * image.height))
                    size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
                    size = (min(size[0], 200 * size[1]), min(size[1], 200 * size[0]))
                    image = image.resize(size, Image.Resampling.LANCZOS)
                output = io.BytesIO()
                image.save(output, format='PNG')
                normalized = output.getvalue()
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ValueError('图片无法安全解码：仅支持静态JPEG/PNG/WEBP、最多2000万像素；请转换或缩小后重试。') from exc
    return {'data_url': 'data:image/png;base64,' + base64.b64encode(normalized).decode('ascii'),
            'source_sha256': hashlib.sha256(data).hexdigest(),
            'normalized_sha256': hashlib.sha256(normalized).hexdigest(),
            'width': image.width, 'height': image.height}


def vision_payload(prompt, data, temperature=0, top_p=1, max_tokens=256):
    if not isinstance(prompt, str) or len(prompt) > 1000:
        raise ValueError('图片问题最多1000个字符；上下文超限时请缩短问题。')
    payload = request_payload(prompt, [], temperature, top_p, max_tokens)
    image = prepare_image(data)
    payload['model'] = 'week5-qwen2-vl-base'
    payload['seed'] = 20260912
    payload['messages'] = [{'role': 'user', 'content': [
        {'type': 'text', 'text': prompt},
        {'type': 'image_url', 'image_url': {'url': image['data_url']}}]}]
    return payload
