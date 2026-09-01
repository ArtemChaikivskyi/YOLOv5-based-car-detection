import re
import sys
from pathlib import Path

import cv2
import torch
import yolov5

# yolov5 7.0.14 викликає torch.load без weights_only=False, а torch >= 2.6
# робить weights_only=True типовим. best.pt — власна натренована модель (довірене
# джерело), тож повертаємо стару поведінку.
_torch_load = torch.load


def _torch_load_full(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _torch_load_full

IMAGES_DIR = Path("images/train")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUT_DIR = Path("runs")


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name.lower())]


def list_images(folder: Path) -> list[Path]:
    imgs = (p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    return sorted(imgs, key=_natural_key)


def detect(model, path: Path):
    """Повертає анотоване зображення (BGR) для показу/збереження."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(rgb)
    annotated = results.render()[0]
    return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)


def main() -> None:
    if not IMAGES_DIR.is_dir():
        raise SystemExit(f"Теку не знайдено: {IMAGES_DIR}")

    images = list_images(IMAGES_DIR)
    if not images:
        raise SystemExit(f"У {IMAGES_DIR} немає зображень")

    # Необовʼязковий аргумент: назва файлу або номер, з якого почати
    idx = 0
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.isdigit():
            idx = int(arg) % len(images)
        else:
            match = [i for i, p in enumerate(images) if p.name == arg or p.stem == arg]
            if match:
                idx = match[0]

    model = yolov5.load("best.pt")
    cache: dict[int, object] = {}

    def annotated(i: int):
        if i not in cache:
            cache[i] = detect(model, images[i])
        return cache[i]

    win = "YOLOv5 Detection"
    headless = False
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except cv2.error:
        headless = True

    if headless:
        # Без дисплея — просто обробити всі й зберегти
        OUT_DIR.mkdir(exist_ok=True)
        for i, path in enumerate(images):
            out = annotated(i)
            if out is not None:
                cv2.imwrite(str(OUT_DIR / f"{path.stem}_detected.jpg"), out)
        print(f"Оброблено {len(images)} зображень -> {OUT_DIR}/")
        return

    print(
        "Керування:  d / → / пробіл — наступне,  a / ← — попереднє,  "
        "s — зберегти,  q / Esc — вихід"
    )
    while True:
        out = annotated(idx)
        title = f"[{idx + 1}/{len(images)}] {images[idx].name}"
        if out is None:
            print(f"Пропущено (не читається): {images[idx].name}")
        else:
            cv2.setWindowTitle(win, title)
            cv2.imshow(win, out)

        key = cv2.waitKeyEx(0)
        if key in (ord("q"), 27):  # q або Esc
            break
        if key in (ord("d"), ord(" "), 32, 63235, 65363, 2555904):  # next / →
            idx = (idx + 1) % len(images)
        elif key in (ord("a"), 63234, 65361, 2424832):  # prev / ←
            idx = (idx - 1) % len(images)
        elif key == ord("s") and out is not None:
            OUT_DIR.mkdir(exist_ok=True)
            dst = OUT_DIR / f"{images[idx].stem}_detected.jpg"
            cv2.imwrite(str(dst), out)
            print(f"Збережено: {dst}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
