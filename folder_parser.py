import os
import json

def create_gabarito_json(root_dir='.'):
    """
    Parses a directory structure to create a nested JSON object.

    Args:
        root_dir (str): The path to the root directory to parse.

    Returns:
        dict: A dictionary representing the nested folder structure and data.
    """
    # Define directories to completely ignore during the walk
    ignored_dirs = {'.venv', '.vscode', '__pycache__'}
    final_json_data = {}

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in ignored_dirs]
        if dirpath == root_dir:
            continue

        path_parts = os.path.relpath(dirpath, root_dir).split(os.sep)
        current_level = final_json_data
        for part in path_parts:
            current_level = current_level.setdefault(part, {})

        if 'data.json' in filenames:
            data_file_path = os.path.join(dirpath, 'data.json')
            try:
                with open(data_file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    current_level.update(content)
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Malformed JSON found in {data_file_path}")
            except IOError as e:
                print(f"️⚠️ Warning: Could not read file {data_file_path}: {e}")

    return final_json_data

if __name__ == "__main__":
    parsed_data = create_gabarito_json()
    output_json_string = json.dumps(parsed_data, indent=2, ensure_ascii=False)

    print("✅ Successfully parsed the directory. Resulting JSON:\n")
    print(output_json_string)

    try:
        with open('data.json', 'w', encoding='utf-8') as f:
            f.write(output_json_string)
        print("\n📄 Output has been saved to data.json")
    except IOError as e:
        print(f"\n❌ Error saving output file: {e}")