import os

# โฟลเดอร์ที่เก็บไฟล์ .cpp / .cu
target_dir = os.path.join(os.getcwd(), "custom_rasterizer")

for root, _, files in os.walk(target_dir):
    for file in files:
        if file.endswith((".cpp", ".cu")):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            if "data_ptr<long>" in content:
                print(f"🔧 Patching {filepath}")
                content = content.replace("data_ptr<long>", "data_ptr<int64_t>")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

print("✅ Patch completed! Now try: pip install --no-build-isolation .")
