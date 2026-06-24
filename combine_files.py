# import os


# def combine_files(root_dir, output_file):
#     # Extensions to include
#     valid_extensions = (
#         ".py",
#         ".js",
#         ".html",
#         ".css",
#         ".md",
#         ".txt",
#         ".json",
#         "Dockerfile",
#         ".yml",
#         ".yaml",
#         ".sh",
#         ".rego",
#         ".toml",
#         ".sql",
#     )

#     # Folders to explicitly ignore
#     ignore_dirs = {".venv", ".git", ".idea", ".vscode", "__pycache__", "node_modules"}

#     with open(output_file, "w", encoding="utf-8") as outfile:
#         for root, dirs, files in os.walk(root_dir):
#             # Modify 'dirs' in-place to prevent os.walk from entering ignored directories
#             dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ignore_dirs]

#             for file in files:
#                 # Skip hidden files
#                 if file.startswith("."):
#                     continue

#                 # Skip the output file and this script itself
#                 if file == output_file or file == "combine_files.py":
#                     continue

#                 # Only process specified file types
#                 if not file.endswith(valid_extensions):
#                     continue

#                 file_path = os.path.join(root, file)
#                 relative_path = os.path.relpath(file_path, root_dir)

#                 # Write the header
#                 outfile.write(f"\n{'=' * 80}\n")
#                 outfile.write(f"FILE: {relative_path}\n")
#                 outfile.write(f"{'=' * 80}\n\n")

#                 # Write content
#                 try:
#                     with open(file_path, encoding="utf-8") as infile:
#                         outfile.write(infile.read())
#                     outfile.write("\n")
#                 except Exception as e:
#                     outfile.write(f"Error reading file: {e}\n")


# if __name__ == "__main__":
#     combine_files(".", "combined_output.txt")
#     print("Files combined successfully into 'combined_output.txt', ignoring hidden items.")
