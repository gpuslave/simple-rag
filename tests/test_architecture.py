import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "rag"
CORE_PATHS = (SOURCE_ROOT / "application", SOURCE_ROOT / "domain")
CONTRACT_PATHS = (SOURCE_ROOT / "ports.py", SOURCE_ROOT / "domain")
FORBIDDEN_CONTRACT_DEPENDENCIES = {
    "fitz",
    "httpx",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langchain_qdrant",
    "langchain_text_splitters",
    "openai",
    "pymupdf",
    "qdrant_client",
    "typer",
}


def python_files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(path.rglob("*.py"))


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_application_core_does_not_import_infrastructure() -> None:
    violations: list[str] = []
    for root in CORE_PATHS:
        for path in python_files(root):
            for module in imported_modules(path):
                if module == "rag.infrastructure" or module.startswith(
                    "rag.infrastructure."
                ):
                    violations.append(f"{path.relative_to(SOURCE_ROOT)}: {module}")

    assert violations == []


def test_domain_and_ports_do_not_expose_technology_dependencies() -> None:
    violations: list[str] = []
    for root in CONTRACT_PATHS:
        for path in python_files(root):
            for module in imported_modules(path):
                if module.split(".", maxsplit=1)[0] in FORBIDDEN_CONTRACT_DEPENDENCIES:
                    violations.append(f"{path.relative_to(SOURCE_ROOT)}: {module}")

    assert violations == []
