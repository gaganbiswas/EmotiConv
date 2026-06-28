import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

REMAP = {0: 0, 4: 1, 5: 1, 2: 2, 3: 3}
DROP = {1}
CLASS_NAMES = ["neutral", "happy", "angry", "sad"]

def convert(in_path: Path, out_path: Path):
    n_dialogs, n_turns, dist = 0, 0, {i: 0 for i in range(4)}
    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            turns = []
            for t in d["turns"]:
                old = int(t["label"])
                if old in DROP:
                    continue
                t = dict(t)
                t["label"] = REMAP[old]
                dist[t["label"]] += 1
                turns.append(t)
            if not turns:
                continue
            d["turns"] = turns
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            n_dialogs += 1
            n_turns += len(turns)
    print(f"{out_path.name}: {n_dialogs} dialogs, {n_turns} turns, "
          f"dist={{ {', '.join(f'{CLASS_NAMES[i]}:{dist[i]}' for i in range(4))} }}")

def main():
    feat = HERE / "features"
    pairs = [
        (feat / "iemocap_features_train.jsonl", feat / "iemocap_features_4class_train.jsonl"),
        (feat / "iemocap_features_test.jsonl",  feat / "iemocap_features_4class_test.jsonl"),
    ]
    for src, dst in pairs:
        if not src.exists():
            raise SystemExit(f"Missing {src}. Run build_features.py + split_features.py first.")
        convert(src, dst)
    print("4-class label map: neutral=0, happy(+excited)=1, angry=2, sad=3  (frustrated dropped)")

if __name__ == "__main__":
    main()
