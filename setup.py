from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def read_requirements() -> tuple[list[str], list[str]]:
    runtime: list[str] = []
    development: list[str] = []
    section = "runtime"

    for raw_line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            if line == "# Development and test dependencies":
                section = "development"
            continue
        if section == "runtime":
            runtime.append(line)
        else:
            development.append(line)
    return runtime, development


runtime_requirements, development_requirements = read_requirements()

setup(
    name="self-healing-rag",
    version="0.1.0",
    description="A self-healing RAG pipeline using LangGraph and FAISS",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["self_healing_rag*"]),
    python_requires=">=3.10",
    install_requires=runtime_requirements,
    extras_require={"dev": development_requirements},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "self-healing-rag=self_healing_rag.cli:main",
        ]
    },
    zip_safe=False,
)
