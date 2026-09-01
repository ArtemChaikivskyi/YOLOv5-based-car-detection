# CAR

YOLOv5-based car detection.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This creates `.venv/` and installs all dependencies (including a managed
Python 3.12 if one isn't already available).

## Run

Гортає зображення з `images/train/` у вікні з детекцією.

```bash
uv run python main.py
```

Почати з конкретного кадру (номер або назва файлу):

```bash
uv run python main.py car50.jpg
```

Керування у вікні: `d` / `→` / пробіл — наступне, `a` / `←` — попереднє,
`s` — зберегти анотований кадр у `runs/`, `q` / `Esc` — вихід.
Без дисплея скрипт обробляє всі зображення й складає їх у `runs/`.

## Dev

```bash
uv run ruff check .
```
