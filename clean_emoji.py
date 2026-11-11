# clean_emoji.py (chạy 1 lần)
import re
from pathlib import Path

REPLACEMENTS = {
    "🚀": "[START]",
    "⚠️": "[WARN]",
    "✅": "[OK]",
    "❌": "[ERROR]",
    "📊": "[CHART]",
    "📈": "[UP]",
    "📉": "[DOWN]",
    "🔬": "[LAB]",
}

def clean_file(path: Path):
    text = path.read_text(encoding="utf-8")
    original = text
    for emoji, repl in REPLACEMENTS.items():
        text = text.replace(emoji, repl)
    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"Cleaned: {path}")

if __name__ == "__main__":
    for p in Path("src").rglob("*.py"):
        clean_file(p)
    for p in Path("scripts").rglob("*.py"):
        clean_file(p)
    clean_file(Path("main.py"))
    print("Done.")