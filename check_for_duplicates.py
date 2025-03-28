import os

# YOLO 라벨 파일이 있는 폴더 경로
label_dir = "dataset/train/labels"

# 모든 라벨 파일 확인
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(label_dir, filename)

        with open(file_path, "r") as f:
            lines = f.readlines()

        # 중복된 라벨을 제거한 후, 새 라벨 리스트
        unique_lines = []
        seen = set()

        for line in lines:
            parts = line.strip().split()
            if parts[0] not in seen:
                unique_lines.append(line.strip())
                seen.add(parts[0])  # 클래스 ID로 중복 확인

        # 중복 제거된 라벨을 파일에 다시 저장
        with open(file_path, "w") as f:
            f.write("\n".join(unique_lines))

print("중복 라벨 제거 완료!")
