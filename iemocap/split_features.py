import json
from pathlib import Path

def get_session(dialog_id):
    return int(dialog_id[3:5])

def split_features(
    input_path,
    train_path=None,
    test_path=None,
    test_session=5,
):
    input_path = Path(input_path)
    if train_path is None:
        train_path = input_path.with_name(f"{input_path.stem}_train.jsonl")
    if test_path is None:
        test_path = input_path.with_name(f"{input_path.stem}_test.jsonl")

    train, test = [], []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            session = get_session(obj["dialog_id"])
            (test if session == test_session else train).append(line)

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train) + "\n")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("\n".join(test) + "\n")

    return len(train), len(test)

def main():
    root = Path(__file__).resolve().parent
    n_train, n_test = split_features(root / "features" / "iemocap_features.jsonl")
    print(f"train: {n_train} test: {n_test}")

if __name__ == "__main__":
    main()
