import os
MAX_CLASS_ID = 11
label_dirs = [
    "D:/PROJECT/DATASET/labels/train",
    "D:/PROJECT/DATASET/labels/val"
]

for label_dir in label_dirs:
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            full_path = os.path.join(label_dir, filename)
            fixed_lines = []

            with open(full_path, "r") as f:
                lines = f.readlines()

            changed = False
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                    if class_id <= MAX_CLASS_ID:
                        fixed_lines.append(line)
                    else:
                        changed = True
                        print(f"❌ {filename}: class {class_id} > {MAX_CLASS_ID} — видалено")
                except ValueError:
                    print(f"⚠️ {filename}: невірний формат — пропущено")

            if changed:
                with open(full_path, "w") as f:
                    f.writelines(fixed_lines)
