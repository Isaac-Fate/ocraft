import typer

from .crnn import crnn_app

app = typer.Typer(
    help="OCRaft is a powerful OCR tool.",
)

# Sub-apps
app.add_typer(crnn_app, name="crnn")


@app.command()
def greet():
    print("Welcome to OCRaft!")
