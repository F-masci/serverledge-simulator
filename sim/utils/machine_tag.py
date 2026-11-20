"""Requisiti sui machine tag delle funzioni, secondo Infrascaler."""

import re


_BASE_ARCHITECTURES = {'x86', 'x86_64', 'amd64', 'arm', 'arm64', 'aarch64'}


def matches_machine_tag(machine_tag: str, pattern: str) -> bool:
    if not pattern:
        return True

    tag = machine_tag.lower()
    full_tag = tag.strip()
    base, *capabilities = full_tag.split('/')
    base = base.strip() or 'unknown'
    arch, _, specialization = base.partition('-')
    base_terms = {full_tag, base, arch.strip() or 'unknown', specialization.strip() or 'default'}
    terms = base_terms | {capability.strip() for capability in capabilities}
    return all(
        _matches_requirement(tag, requirement.strip(), base_terms, terms)
        for requirement in pattern.lower().split(',')
    )


def _matches_requirement(
    tag: str, requirement: str, base_terms: set[str], terms: set[str],
) -> bool:
    if not requirement:
        return True
    if requirement.startswith('^') or requirement.endswith('$') or '.*' in requirement:
        try:
            return re.search(requirement, tag) is not None
        except (re.PatternError, OverflowError):
            return False
    if '*' in requirement:
        wildcard = re.escape(requirement).replace(r'\*', '.*')
        return re.fullmatch(wildcard, tag) is not None

    first, *rest = (part.strip() for part in requirement.split('/'))
    allowed = base_terms if rest and ('-' in first or first in _BASE_ARCHITECTURES) else terms
    return (not first or first in allowed) and all(not term or term in terms for term in rest)
