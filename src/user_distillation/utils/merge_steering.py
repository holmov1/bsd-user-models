import json
from pathlib import Path
def merge_steering_files(file_paths: list[str | Path], output_path: str | Path):
    merged_data = {'_meta': {}}
    for file_path in file_paths:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, "r") as f:
            data = json.load(f)
        if len(merged_data['_meta'].keys()) < 1:
            merged_data['_meta'] = data.get('_meta')

        assert merged_data['_meta']['alphas'] == data.get('_meta').get('alphas')
        data.pop('_meta', None)
        for key, value in data.items():
            if key not in merged_data:
                merged_data[key] = data[key]
    with open(output_path, "w") as f:
        json.dump(merged_data, f, indent=2)
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge steering files into one.")
    parser.add_argument("--paths", nargs="+", help="Paths to the steering files to merge.")
    parser.add_argument("--out", required=True, help="Path to save the merged output file.")
    args = parser.parse_args()

    merge_steering_files(args.paths, args.out)
        
        
        