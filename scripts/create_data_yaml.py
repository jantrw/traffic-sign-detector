# one-off: make_data_yaml.py
from pathlib import Path
import yaml
classes = Path("data/yolo_signs/classes.txt").read_text(encoding="utf8").splitlines()
data = {
  "path": "data/yolo_signs",
  "train": "train/images",
  "val": "val/images",
  "nc": len(classes),
  "names": {i: name for i, name in enumerate(classes)}
}
Path("data/yolo_signs/data.yaml").write_text(yaml.dump(data, sort_keys=False, allow_unicode=True), encoding="utf8")
