"""Build and install the fla_npu AscendC fused KDA ops.

Mirrors ``xllm/compiler/tilelang/tilelang_ascend_install.py``: an
``ensure_*_ready`` / detect-then-install entry point the xLLM build calls for
the NPU device. The build is skipped entirely when the active environment
already imports the KDA ops.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

from scripts.build_support.env import set_npu_envs
from scripts.build_support.utils import get_ascend_platform, get_cmake_dir
from scripts.logger import logger

# fla_npu commit that landed ``recurrent_kda`` (PR #266). The decode path needs
# a tree at least this new; older trees only expose ``chunk_kda_fwd``.
FLA_NPU_KDA_COMMIT = "f289537843e83ee8a6a2e09bb83828481b5c84d0"
FLA_NPU_REPO = "https://github.com/flashserve/flash-linear-attention-npu.git"

# KDA fused ops fla_npu must expose for KDA to run.
FLA_NPU_KDA_OPS = ("chunk_kda_fwd", "recurrent_kda", "causal_conv1d")

# xllm platform code (get_ascend_platform: a2/a3/a5) -> fla_npu FLA_NPU_SOC.
# Kept in lockstep with tilelang's platform detection so both compile for the
# same chip family.
_FLA_NPU_SOC = {
    "a2": "ascend910b",
    "a3": "ascend910_93",
    "a5": "ascend950",
}

PREPARE_FLA_NPU_COMMAND = (
    "python setup.py bdist_wheel --device=npu"
    "  # _maybe_install_fla_npu fetches/builds the KDA ops automatically;"
    " FLA_NPU_SRC=<dir> FLA_NPU_SOC=<ascend910b|ascend910_93|ascend950> override"
)


def fla_npu_ready() -> bool:
    """Return True if fla_npu imports and exposes all KDA fused ops."""
    check = (
        "import fla_npu; from fla_npu.ops.ascendc import "
        + ", ".join(FLA_NPU_KDA_OPS)
    )
    result = subprocess.run(
        [sys.executable, "-c", check],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _resolve_soc() -> str:
    """Map xllm's Ascend platform code to fla_npu's FLA_NPU_SOC.

    Honors an explicit FLA_NPU_SOC env override; otherwise maps
    get_ascend_platform() (a2/a3/a5) to the fla_npu SoC family.
    """
    override = os.environ.get("FLA_NPU_SOC", "").strip()
    if override:
        return override
    return _FLA_NPU_SOC.get(get_ascend_platform(), "ascend910b")


def _resolve_source_tree() -> tuple[str, bool]:
    """Locate the fla_npu source tree to build.

    Returns (path, cloned_here). Honors an explicit FLA_NPU_SRC env path;
    otherwise clones into the CMake build dir so the tree is reused across
    rebuilds.
    """
    src = os.environ.get("FLA_NPU_SRC", "").strip()
    if src and os.path.isdir(os.path.join(src, "setup.py")):
        return src, False
    return os.path.join(get_cmake_dir(), "fla_npu_src"), True


def _checkout_kda_commit(src: str, env: dict[str, str], cloned_here: bool) -> None:
    """Fetch and check out the KDA commit in the source tree."""
    if cloned_here:
        logger.info(f"initializing fla_npu source tree at {src}")
        os.makedirs(src, exist_ok=True)
        subprocess.check_call(["git", "init", src], env=env)
        subprocess.check_call(
            ["git", "remote", "add", "origin", FLA_NPU_REPO],
            cwd=src, env=env,
        )
    logger.info(f"fetching fla_npu KDA commit {FLA_NPU_KDA_COMMIT}")
    subprocess.check_call(
        ["git", "fetch", "--depth", "1", "origin", FLA_NPU_KDA_COMMIT],
        cwd=src,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        ["git", "checkout", "FETCH_HEAD"], cwd=src, env=env
    )


def _remove_stale_libopapi() -> None:
    """Remove the stale libopapi.so older fla_npu wheels ship alongside
    libcust_opapi.so; it shadows the CANN runtime and breaks tiling."""
    site_pkg = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import fla_npu, os; print(os.path.dirname(fla_npu.__file__))",
        ],
        text=True,
    ).strip()
    stale = os.path.join(
        site_pkg,
        "opp",
        "vendors",
        "fla_npu_transformer",
        "op_api",
        "lib",
        "libopapi.so",
    )
    if os.path.exists(stale):
        os.remove(stale)
        logger.info("removed stale fla_npu libopapi.so")


def prepare_fla_npu(*, force: bool = False) -> Path | None:
    """Ensure the fla_npu AscendC fused KDA ops are importable.

    No-op (unless ``force``) when the active environment already imports them.
    Otherwise the fla_npu source tree is fetched, pinned to the commit that
    landed ``recurrent_kda``, built into a wheel (~3.5 min, no incremental
    benefit — the kernel rebuilds every time), and force-installed.

    The runtime OPP hook (sourcing fla_npu's ``set_env.bash`` before any
    torch.npu op, to avoid aclnn ``561103``) stays a launch-script concern; this
    function only builds and installs the wheel.
    """
    set_npu_envs()

    if not force and fla_npu_ready():
        logger.info("fla_npu already installed with KDA ops; skip build")
        return None

    src, cloned_here = _resolve_source_tree()
    soc = _resolve_soc()
    env = os.environ.copy()
    env["FLA_NPU_SOC"] = soc

    _checkout_kda_commit(src, env, cloned_here)

    dist_dir = os.path.join(src, "dist")
    os.makedirs(dist_dir, exist_ok=True)
    logger.info(f"building fla_npu wheel (FLA_NPU_SOC={soc}); ~3.5 min")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "-w",
            dist_dir,
            ".",
        ],
        cwd=src,
        env=env,
    )
    wheels = glob.glob(
        os.path.join(dist_dir, "flash_linear_attention_npu-*.whl")
    )
    if not wheels:
        raise RuntimeError("fla_npu wheel build produced no artifact")
    logger.info(f"installing fla_npu wheel: {os.path.basename(wheels[0])}")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            wheels[0],
        ],
        env=env,
    )

    _remove_stale_libopapi()

    if not fla_npu_ready():
        raise RuntimeError(
            "fla_npu install finished but KDA ops still not importable"
        )
    logger.info("fla_npu KDA ops installed successfully")
    return None


def ensure_fla_npu_ready() -> None:
    """Raise if fla_npu KDA ops are not importable after attempting install."""
    if not fla_npu_ready():
        raise RuntimeError(
            "fla_npu KDA ops are not importable.\n"
            f"Run `{PREPARE_FLA_NPU_COMMAND}` or rebuild with --device=npu."
        )
