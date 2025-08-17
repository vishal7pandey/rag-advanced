from __future__ import annotations
import typer

app = typer.Typer()


@app.command()
def main(qa: str = typer.Option(..., help="QA yaml"), flow: str = typer.Option("hybrid")):
    typer.echo(f"Offline eval stub: qa={qa} flow={flow}")


if __name__ == "__main__":
    app()
