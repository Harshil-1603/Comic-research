import cv2, os, pandas as pd

rows = []
for f in sorted(os.listdir("data/processed")):
    p = os.path.join("data/processed", f)
    img = cv2.imread(p)
    cv2.imshow("panel", img)
    cv2.waitKey(1)
    e = input("emotion [anger/sadness/joy/fear/neutral]: ").strip()
    t = input("text (optional): ").strip()
    rows.append([f, e, t, "unknown"])
    cv2.destroyAllWindows()

pd.DataFrame(rows, columns=["image", "emotion", "text", "source"]).to_csv(
    "data/annotations.csv", index=False
)
