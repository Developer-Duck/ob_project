import os

# YOLO 라벨 파일이 있는 폴더 경로
label_dir = "dataset/detect can.v1i.yolov11/valid/labels"

for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(label_dir, filename)

        with open(file_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            parts[0] = "2"  # 모든 클래스 ID를 1로 변경

            new_lines.append(" ".join(parts))

        with open(file_path, "w") as f:
            f.write("\n".join(new_lines))

print("YOLO 라벨 클래스 ID 변경 완료!")
