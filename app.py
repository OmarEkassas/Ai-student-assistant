"""Entry point. Run `python app.py` to launch the UI."""

from __future__ import annotations

from src.ui.app import build_ui


def main() -> None:
    demo = build_ui()
    demo.queue(default_concurrency_limit=4)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
    )


if __name__ == "__main__":
    main()