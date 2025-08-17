from __future__ import annotations
from pathlib import Path
import typer

app = typer.Typer()


@app.command()
def main(
    paths: str = typer.Option("data/samples", help="Directory to ingest"),
    chunk_size: int = 800,
    overlap: int = 120,
):
    base = Path(paths)
    if not base.exists():
        typer.echo(f"Path not found: {base}")
        raise typer.Exit(code=1)
    # Placeholder: actual ingest to be implemented
    typer.echo(f"Ingesting from {base} with chunk_size={chunk_size} overlap={overlap}")


if __name__ == "__main__":
    app()
