"""Draw the fixed, seeded COCO val2017 subsets.

Writes
  data/subset/image_ids.txt   evaluation frames (n_images)
  data/subset/attack_ids.txt  disjoint images used only to optimise the perturbation
Usage:  python src/make_subset.py
"""
import json
import os
import random

from common import load_config, p


def main():
    cfg = load_config()
    s = cfg["subset"]
    ann = p(cfg["paths"]["coco_annotations"])
    if os.path.exists(ann):
        with open(ann) as f:
            ids = sorted(img["id"] for img in json.load(f)["images"])
    else:  # fall back to the image folder listing
        ids = sorted(int(n.split(".")[0]) for n in os.listdir(p(cfg["paths"]["coco_images"])) if n.endswith(".jpg"))
    rng = random.Random(s["seed"])
    rng.shuffle(ids)
    n_eval, n_atk = s["n_images"], s["n_attack_train"]
    if len(ids) < n_eval + n_atk:
        raise SystemExit("only %d images found; need %d" % (len(ids), n_eval + n_atk))
    out = os.path.dirname(p(cfg["paths"]["subset_ids"]))
    os.makedirs(out, exist_ok=True)
    with open(p(cfg["paths"]["subset_ids"]), "w") as f:
        f.write("\n".join(str(i) for i in sorted(ids[:n_eval])) + "\n")
    with open(os.path.join(out, "attack_ids.txt"), "w") as f:
        f.write("\n".join(str(i) for i in sorted(ids[n_eval:n_eval + n_atk])) + "\n")
    print("eval subset: %d images, attack-train: %d images (seed %d, disjoint)" % (n_eval, n_atk, s["seed"]))


if __name__ == "__main__":
    main()
