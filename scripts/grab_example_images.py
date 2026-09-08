"""Copy two representative CXRs from the Kaggle input tree into figs/.

Run this as a cell inside the Kaggle notebook after attaching the CheXpert dataset.
The images are already mounted read-only under /kaggle/input — this script only
copies them to ./figs/ so make_architecture.py can load them as local files.

We build the image paths directly from known patient/study/view IDs rather than
doing a recursive glob over the ~200 k-file tree; a full os.walk on that tree is
what makes naive approaches feel slow on Kaggle.

After this runs, download the two files from the Kaggle output panel and place
them in your local figs/ directory so make_architecture.py can find them.
"""
import os
import shutil

# Patient/study/view combos chosen to match the figure caption in the paper:
#   edema_pos  — "88 years Female SOB, eval for pulmonary edema" → label positive
#   pneumonia_neg — "66Y/F with NSCLC, productive cough x 5 days r/o PNA" → label negative
WANT = {
    "example_edema_pos.jpg":     "train/patient24238/study3/view1_frontal.jpg",
    "example_pneumonia_neg.jpg": "train/patient26588/study3/view1_frontal.jpg",
}


def find_image_root(base="/kaggle/input", max_depth=6):
    """Return the directory that directly contains `train/`.

    Walks down from `base` checking directory names only (no file listing)
    and stops at max_depth so it stays fast on the large Kaggle input tree.
    """
    for root, dirs, _ in os.walk(base):
        depth = root[len(base):].count(os.sep)
        if depth >= max_depth:
            dirs[:] = []   # prune: don't descend further
            continue
        if "train" in dirs:
            return root
    raise SystemExit(
        "Could not find a 'train/' folder under /kaggle/input — "
        "check os.listdir('/kaggle/input/datasets/ashery/chexpert')."
    )


def main():
    root = find_image_root()
    print("image root:", root)

    os.makedirs("figs", exist_ok=True)
    for out_name, rel_path in WANT.items():
        src = os.path.join(root, rel_path)
        if not os.path.exists(src):
            print(f"!! not found: {src}")
            continue
        dst = f"figs/{out_name}"
        shutil.copy(src, dst)
        print(f"ok  {src}  ->  {dst}")

    print(
        "\nDownload figs/example_edema_pos.jpg and figs/example_pneumonia_neg.jpg "
        "from the Kaggle output panel, and drop them into your local figs/."
    )


if __name__ == "__main__":
    main()
