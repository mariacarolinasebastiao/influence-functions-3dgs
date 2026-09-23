"""Filesystem roots, read from the environment so the scripts are not tied to one machine.

GS_WORK_ROOT        nerfstudio projects, checkpoints and LOO datasets
GS_PROJ_ROOT        where results were written; on the original setup this was a separate
                    shared filesystem, so it is configurable on its own and falls back to
                    GS_WORK_ROOT
GS_NERFSTUDIO_REPO  the nerfstudio checkout the pipeline stages chdir into
"""
import os

WORK_ROOT = os.environ.get("GS_WORK_ROOT", "/home/coder/data")
PROJ_ROOT = os.environ.get("GS_PROJ_ROOT", WORK_ROOT)
NERFSTUDIO_REPO = os.environ.get("GS_NERFSTUDIO_REPO", f"{PROJ_ROOT}/nerfstudio/nerfstudio")
