import cv2
from pathlib import Path


def preprocess_img(img_path, output_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Cannot read image: {img_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11
    )
    final = cv2.medianBlur(binary, 5)
    cv2.imwrite(str(output_path), final)
    return True

def segment_word(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    image_extensions = {". jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    img_paths = [
        p for p in input_folder.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    if len(img_paths) == 0:
        print("No images found in input folder.")
        return 

    print(f"Found {len(img_paths)} images.")

    for img_path in img_paths:
        output_path = output_folder / img_path.name
        success = preprocess_img(img_path, output_path)

        if success:
            print(f"[OK] {img_path.name} -> {output_path.name}")

    print("[DONE] Finished preprocessing folder.")
