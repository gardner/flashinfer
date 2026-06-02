from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text()


def workflow_header(path: str) -> str:
    return "\n".join(read(path).splitlines()[:24])


def test_gb10_release_workflow_builds_native_sm121a_jit_cache():
    workflow = read(".github/workflows/gb10-release.yml")

    assert "gb10-flashinfer-v*" in workflow
    assert 'local_version="${cuda_suffix}gb10"' in workflow
    assert workflow.count(
        "FLASHINFER_LOCAL_VERSION: ${{ needs.setup.outputs.local_version }}"
    ) == 3
    assert 'FLASHINFER_CUDA_ARCH_LIST: "12.1a"' in workflow
    assert "manylinuxaarch64-builder:cuda" in workflow
    assert "flashinfer-jit-cache" in workflow
    assert "cuobjdump" in workflow
    assert "SHA256SUMS" in workflow
    assert "release-metadata.json" in workflow
    assert "--notes-file dist/release-notes.md" in workflow
    assert "gh release upload" in workflow


def test_broad_release_automation_is_opt_in_for_the_fork():
    assert "schedule:" not in workflow_header(".github/workflows/nightly-release.yml")

    docker_release_header = workflow_header(".github/workflows/release-ci-docker.yml")
    assert "workflow_dispatch:" in docker_release_header
    assert "push:" not in docker_release_header
    assert "pull_request:" not in docker_release_header
