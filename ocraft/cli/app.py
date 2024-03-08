import typer

app = typer.Typer(
    help="OCRaft is a powerful OCR tool.",
)


@app.command()
def greet():
    print("Welcome to OCRaft!")
