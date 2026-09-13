"""The CLI flags every `bench_*.py` script shares (PLAN-V2 §4.2, W0.4):

  --device cpu|mps   which device to run this invocation's cases on
  --only <substr>    filter cases by substring match on case id
  --full             opt into real sizes and the >= 10 sample floor;
                      default is a smoke run (tiny sizes, 1 warm-up,
                      2 timed samples) — see bench/README.md.

An unknown flag exits non-zero naming the flag, the same "fail loudly"
rule `bench/lib/cli.ts` applies to the TypeScript side.
"""
import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=["cpu", "mps"], required=True)
    parser.add_argument("--only", default=None)
    parser.add_argument("--full", action="store_true")
    return parser.parse_args(argv)
