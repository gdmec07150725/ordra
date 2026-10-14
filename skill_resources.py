"""Bounded UTF-8 Markdown references declared by a local skill."""
import hashlib
import re


MAX_RESOURCE_READS = 4
MAX_REFERENCES = 8
MAX_RESOURCE_BYTES = 24000
RESOURCE_TOOL = {'type': 'function', 'function': {
    'name': 'read_skill_resource',
    'description': '读取已加载技能列出的 references 文件。仅在当前情况需要细节时单独调用；不是业务查询或证据，不执行文件。',
    'parameters': {'type': 'object', 'properties': {
        'name': {'type': 'string', 'maxLength': 64},
        'path': {'type': 'string', 'maxLength': 200}},
        'required': ['name', 'path'], 'additionalProperties': False}}}


def valid_reference(path):
    return (isinstance(path, str) and len(path) <= 200 and path.startswith('references/')
            and path.endswith('.md') and not re.search(r'[\\:\x00-\x20<>"|?*%#]', path)
            and all(part not in ('', '.', '..') and not part.endswith(('.', ' '))
                    for part in path.split('/')))


def read_reference(root, name, path):
    if not valid_reference(path):
        raise ValueError('无效的引用文件路径')
    base = (root / name).resolve()
    target = (base / path).resolve()
    if not base.is_relative_to(root) or not target.is_relative_to(base / 'references'):
        raise ValueError('引用文件必须位于当前技能 references 目录内')
    # Reject links/junctions even when they currently resolve inside the skill.
    part = root / name
    for segment in ('', *path.split('/')):
        if segment:
            part /= segment
        if part.is_symlink() or part.is_junction():
            raise ValueError('引用文件不允许符号链接或目录联接')
    with target.open('rb') as source:
        raw = source.read(MAX_RESOURCE_BYTES + 1)
    if len(raw) > MAX_RESOURCE_BYTES:
        raise ValueError('引用文件过大')
    content = raw.decode('utf-8-sig')
    if not content.strip():
        raise ValueError('引用文件为空')
    return {'path': path, 'version': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}, content


def reference_manifest(root, name, body):
    paths = list(dict.fromkeys(re.findall(r'\]\((references/[^\s)]+)\)', body)))
    if len(paths) > MAX_REFERENCES:
        raise ValueError('技能最多声明 8 个引用文件')
    manifest = []
    for path in paths:
        if not valid_reference(path):
            raise ValueError('技能引用路径无效')
        try:
            metadata, _ = read_reference(root, name, path)
        except (OSError, ValueError, UnicodeError):
            metadata = {'path': path, 'version': None, 'unavailable': True}
        manifest.append(metadata)
    return manifest
