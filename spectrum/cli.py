"""Library commands. Dated experiments and reports live in the archive branch."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("families", help="List causal families and their native input contracts")
    catalog.add_argument("--group")
    ingest = commands.add_parser("ingest", help="Partition DBN only at explicit source resets")
    ingest.add_argument("--source", required=True)
    ingest.add_argument("--output", required=True)
    ingest.add_argument("--route", nargs=3, type=int, required=True)
    ingest.add_argument("--member")
    build = commands.add_parser("build", help="Build canonical market tapes with the shared MBO kernel")
    build.add_argument("--raw", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--workers", type=int, default=4)
    features = commands.add_parser("features", help="Materialize frame families from canonical seconds")
    features.add_argument("--market", required=True)
    features.add_argument("--output", required=True)
    features.add_argument("--families", nargs="+", default=["motion", "flow", "spectral"])
    features.add_argument("--plugin", action="append", default=[])
    features.add_argument("--workers", type=int, default=4)
    replay = commands.add_parser("replay", help="Read a finalized market publication")
    replay.add_argument("capture")
    replay.add_argument("--rows", type=int, default=5)
    replay.add_argument("--layer", default="quotes")
    bank = commands.add_parser("bank", help="Build one crossing bank per day (bid OR ask rule)")
    bank.add_argument("--days", required=True, help="Folder of <month>/<date>/manifest.json day manifests")
    bank.add_argument("--output", required=True)
    bank.add_argument("--workers", type=int, default=4)
    verify = commands.add_parser("verify-market", help="Verify all finalized payload hashes")
    verify.add_argument("capture")
    args = parser.parse_args(argv)
    if hasattr(args, "workers") and not 1 <= args.workers <= 12:
        parser.error("workers must be between 1 and 12")
    if args.command == "families":
        from .features.catalog import families
        result = [f.describe() for f in families(args.group)]
    elif args.command == "ingest":
        from .market.ingest import partition
        result = partition(args.source, args.output, args.route, args.member)
    elif args.command == "build":
        from .market.build import build
        result = build(args.raw, args.output, args.workers)
    elif args.command == "features":
        from .features import generate
        result = generate(args.market, args.output, args.families, args.workers, args.plugin)
    elif args.command == "bank":
        from .crossings.build import build_banks
        result = build_banks(args.days, args.output, args.workers)
    elif args.command == "verify-market":
        from .market.finalized import verify
        result = verify(args.capture)
    else:
        if args.rows <= 0:
            parser.error("rows must be positive")
        from .market import FinalizedReplay
        stream = FinalizedReplay(args.capture, layer=args.layer)
        table = stream.read_batch(args.rows)
        result = {"rows": [] if table is None else table.to_pylist(), "cursor": stream.checkpoint()}
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
