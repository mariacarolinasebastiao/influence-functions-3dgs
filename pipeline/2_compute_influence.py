import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

from src.fim_influence import generate_influence_files
from configs.scenes import build_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="garden", choices=["garden", "bicycle", "room"])
    ap.add_argument("--no-dot", action="store_true", help="skip the raw dot-product baseline files")
    a = ap.parse_args()

    S = build_scene(a.scene, build_fim=False)
    generate_influence_files(S.module, S.results_dir, S.TESTS, S.GROUPS, do_dot=not a.no_dot)


if __name__ == "__main__":
    main()
