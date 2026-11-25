"""Verifica dei requisiti sui machine tag delle funzioni, secondo Infrascaler."""

import re

# Architetture di base riconosciute nella prima componente del machine tag.
_BASE_ARCHITECTURES = {'x86', 'x86_64', 'amd64', 'arm', 'arm64', 'aarch64'}


##################
### Public API ###
##################

# Costrutti PCRE/Python non supportati dal motore RE2 di Go (lookaround, atomic group e backreference).
# Nota: altre sottigliezze RE2 (es. classi POSIX [[:alpha:]]) restano legate al motore regex Python.
_UNSUPPORTED_RE2 = re.compile(r'(?<!\\)(?:\\\\)*\(\?(?:=|!|<=|<!|>)|(?<!\\)(?:\\\\)*\\[1-9]')


def validate_tag_pattern(pattern: str) -> None:
    """Valida il pattern secondo normalizeTagPatternForAPI di Go (intero pattern grezzo + RE2)."""
    if not pattern:
        return
    if pattern.startswith('^') or pattern.endswith('$') or '.*' in pattern:
        if _UNSUPPORTED_RE2.search(pattern) is not None:
            raise ValueError(f'tag_pattern non supportato da RE2: {pattern!r}')
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f'tag_pattern non valido: {pattern!r}') from error


def matches_machine_tag(machine_tag: str, pattern: str) -> bool:
    """Verifica che il machine tag soddisfi tutti i requisiti separati da virgola nel pattern."""
    if not pattern:
        return True

    # Normalizza il tag e scompone la parte base (architettura-specializzazione)
    # dalle capability separate da '/'.
    tag = machine_tag.lower()
    full_tag = tag.strip()
    base, *capabilities = full_tag.split('/')
    base = base.strip() or 'unknown'
    arch, _, specialization = base.partition('-')

    # Costruisce i termini base e l'insieme completo di termini confrontabili.
    base_terms = {full_tag, base, arch.strip() or 'unknown', specialization.strip() or 'default'}
    terms = base_terms | {capability.strip() for capability in capabilities}

    # Tutti i requisiti elencati nel pattern (AND logico) devono risultare soddisfatti.
    return all(
        _matches_requirement(tag, requirement.strip(), base_terms, terms)
        for requirement in pattern.lower().split(',')
    )


########################
### Matching helpers ###
########################

def _matches_requirement(
    tag: str, requirement: str, base_terms: set[str], terms: set[str],
) -> bool:
    """Valuta un singolo requisito (regex, wildcard o confronto per termini separati da '/')."""
    if not requirement:
        return True

    # Espressione regolare esplicita (^, $, .*).
    if requirement.startswith('^') or requirement.endswith('$') or '.*' in requirement:
        try:
            return re.search(requirement, tag) is not None
        except (re.error, OverflowError):
            return False

    # Pattern con wildcard '*' tradotto in corrispondenza sull'intera stringa.
    if '*' in requirement:
        wildcard = re.escape(requirement).replace(r'\*', '.*')
        return re.fullmatch(wildcard, tag) is not None

    # Confronto per termini separati da '/': il primo termine può vincolare la sola parte base.
    first, *rest = (part.strip() for part in requirement.split('/'))
    allowed = base_terms if rest and ('-' in first or first in _BASE_ARCHITECTURES) else terms
    return (not first or first in allowed) and all(not term or term in terms for term in rest)
