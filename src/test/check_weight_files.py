from pathlib import Path

def find_model_files(root_dir: str):
    root = Path(root_dir)

    # よくあるモデル拡張子
    model_exts = [
        ".pt", ".pth", ".bin", ".ckpt",
        ".safetensors", ".onnx", ".engine"
    ]

    results = []

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in model_exts:
            size_mb = path.stat().st_size / (1024 * 1024)

            results.append({
                "path": str(path),
                "size_mb": round(size_mb, 2)
            })

    return results


def print_models(root_dir: str):
    models = find_model_files(root_dir)

    if not models:
        print("モデルファイルは見つかりませんでした")
        return

    print(f"=== 見つかったモデル: {len(models)}件 ===")
    for m in models:
        print(f"{m['size_mb']:8.2f} MB  |  {m['path']}")
        

# テスト実行
print_models("/workspace/third_party/")        