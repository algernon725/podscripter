#!/usr/bin/env python3
"""
Guard against the class of defect found in v0.12.0 and fixed in v0.12.1.

`_transformer_based_restoration()` had an early `return` inside its
`if language == 'es':` block, leaving the following 242 lines permanently
unreachable. The suite stayed green the whole time, because the dead region's
helpers were unit-tested directly while never running in production.

This test is the AST scan that found it, kept as a permanent check:
no statement may follow a terminating statement inside the same block.

It also verifies that every module-level function in the core modules has at
least one caller, so a helper cannot quietly become an orphan again.
"""

import ast
import pathlib

import pytest

pytestmark = pytest.mark.core

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Modules owned by this project (excludes tests, which are collected separately).
SOURCE_FILES = [
    "podscripter.py",
    "punctuation_restorer.py",
    "sentence_splitter.py",
    "sentence_formatter.py",
    "speaker_diarization.py",
    "domain_utils.py",
    "language_support.py",
]

TERMINATORS = (ast.Return, ast.Raise, ast.Break, ast.Continue)

# Blocks that a terminating statement ends.
BLOCK_FIELDS = ("body", "orelse", "finalbody")


def _iter_blocks(tree):
    """Yield (node, field_name, statement_list) for every statement block."""
    for node in ast.walk(tree):
        for field in BLOCK_FIELDS:
            block = getattr(node, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield node, field, block


@pytest.mark.parametrize("filename", SOURCE_FILES)
def test_no_statements_after_a_terminator(filename):
    """No code may sit after a return/raise/break/continue in the same block."""
    path = REPO_ROOT / filename
    tree = ast.parse(path.read_text())

    unreachable = []
    for node, field, block in _iter_blocks(tree):
        for index, stmt in enumerate(block[:-1]):
            if isinstance(stmt, TERMINATORS):
                dead = block[index + 1:]
                unreachable.append(
                    f"{filename}:{stmt.lineno} — {type(stmt).__name__} in "
                    f"{type(node).__name__}.{field} leaves lines "
                    f"{dead[0].lineno}-{dead[-1].end_lineno} unreachable "
                    f"({len(dead)} statements)"
                )

    assert not unreachable, "Unreachable code found:\n  " + "\n  ".join(unreachable)


def _module_level_functions(tree):
    return {
        node.name: node.lineno
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _referenced_names():
    """Every name referenced anywhere in the project's Python files.

    Attribute access and string literals are included so that `module.helper`,
    `__all__` entries and any `getattr`-style lookup count as a reference.
    """
    referenced = set()
    files = [REPO_ROOT / name for name in SOURCE_FILES]
    files += sorted(REPO_ROOT.glob("tests/**/*.py"))
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                referenced.add(node.value)
    return referenced


@pytest.mark.parametrize("filename", SOURCE_FILES)
def test_no_orphaned_module_level_functions(filename):
    """Every module-level function must be called, exported, or tested."""
    tree = ast.parse((REPO_ROOT / filename).read_text())
    referenced = _referenced_names()

    orphans = [
        f"{filename}:{lineno} — {name}()"
        for name, lineno in _module_level_functions(tree).items()
        if name not in referenced
    ]

    assert not orphans, (
        "Module-level functions with no caller anywhere (delete them, or wire "
        "them up):\n  " + "\n  ".join(orphans)
    )
