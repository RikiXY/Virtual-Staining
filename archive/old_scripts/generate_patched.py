import cv2, os, sys


def main(path: str):
    lf = cv2.imread(f"{path}/label_free.tif")[200:-200, 200:-200]

    generated = []
    for name in os.listdir(f"{path}/generated"):
        img = cv2.imread(f"{path}/generated/{name}")
        generated.append((img, name))

    for img, name in generated:
        x, y = map(int, name.split("_")[:2])
        h, w = img.shape[:2]
        lf[y:y+h, x:x+w] = img
    cv2.imwrite(f"{path}/patched.tif", lf)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python generate_patched.py <path>")
        sys.exit(1)
    main(sys.argv[1])