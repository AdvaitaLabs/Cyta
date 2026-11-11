"""Command-line entry points for Advaita_VirtualCell_Model."""

import argparse
import json

from . import MODEL_NAME, __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="advaita", description=MODEL_NAME)
    parser.add_argument("--version", action="version", version=f"{MODEL_NAME} {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("infer", "Export trained model readouts"),
        ("train", "Adapt source amplitude, bulk or specialist readouts"),
        ("train-foundation", "Train all foundation modules from random initialization"),
        ("prepare-source", "Build pooled source response archives"),
        ("prepare-specialist", "Compute arithmetic labels and precision weights"),
        ("fit-basis", "Fit the source-only common response basis"),
        ("fit-router", "Build control-only background references"),
        ("evaluate", "Compute count-matched local diagnostics"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True)
    render = commands.add_parser("render", help="Render response readouts into count blocks")
    render.add_argument("--config", required=True)
    render.add_argument(
        "--indices", type=int, nargs="+", help="Original target-list indices to render"
    )
    render.add_argument("--workers", type=int, default=1)
    assembly = commands.add_parser("assemble", help="Assemble verified sparse blocks into H5AD")
    assembly.add_argument("--manifest", required=True)
    assembly.add_argument("--output", required=True)
    validate = commands.add_parser("validate", help="Check raw counts, axes and target coverage")
    validate.add_argument("--input", required=True)
    validate.add_argument("--targets")
    validate.add_argument("--genes")
    validate.add_argument("--contexts", nargs="+")
    validate.add_argument("--cells-per-target", type=int, default=400)
    demo = commands.add_parser("demo", help="Run the complete rendering pipeline on synthetic data")
    demo.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import run_demo

            result = run_demo(args.output)
        elif args.command == "render":
            from .pipeline import render_from_config

            result = {
                "manifest": str(
                    render_from_config(args.config, indices=args.indices, workers=args.workers)
                )
            }
        elif args.command == "assemble":
            from .assembly import assemble

            result = assemble(args.manifest, args.output)
        elif args.command == "validate":
            from .validation import validate_h5ad

            result = validate_h5ad(
                args.input,
                targets=args.targets,
                genes=args.genes,
                contexts=args.contexts,
                cells_per_target=args.cells_per_target,
            )
        elif args.command == "infer":
            from .inference import infer_from_config

            result = {"manifest": str(infer_from_config(args.config))}
        elif args.command == "train":
            from .training.runner import train_from_config

            result = train_from_config(args.config)
        elif args.command == "train-foundation":
            from .training.foundation import train_foundation

            result = train_foundation(args.config)
        elif args.command == "evaluate":
            from .evaluation import evaluate_from_config

            result = evaluate_from_config(args.config)
        else:
            from . import preparation

            operation = {
                "prepare-source": preparation.prepare_source,
                "prepare-specialist": preparation.prepare_specialist,
                "fit-basis": preparation.fit_basis,
                "fit-router": preparation.fit_router,
            }[args.command]
            result = operation(args.config)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as error:
        parser.exit(2, f"advaita: {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
