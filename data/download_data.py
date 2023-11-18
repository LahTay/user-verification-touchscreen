from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import json
import uuid
import re

# TODO: READ THE README FILE BEFORE RUNNING THIS SCRIPT


def authenticate():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)


def read_processed_files(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return set(file.read().splitlines())
    return set()


def add_to_processed_files(file_path, filename):
    with open(file_path, "a") as file:
        file.write(filename + "\n")


def generate_unique_number(name, name_to_number, used_numbers):
    if name not in name_to_number:
        new_number = max(used_numbers, default=0) + 1
        name_to_number[name] = new_number
        used_numbers.add(new_number)
        with open(name_to_number_path, "w") as f:
            json.dump(name_to_number, f)
    return name_to_number[name]


def download_and_rename_files(
    drive,
    folder_id,
    local_path,
    processed_files,
    processed_files_path,
    name_to_number_path,
):
    os.makedirs(local_path, exist_ok=True)
    file_list = drive.ListFile(
        {"q": f"'{folder_id}' in parents and trashed=false"}
    ).GetList()
    past_number = 0
    # Load the mapping of names to numbers and the set of used numbers
    if os.path.isfile(name_to_number_path):
        with open(name_to_number_path, "r") as f:
            name_to_number = json.load(f)
    else:
        name_to_number = {}

    used_numbers = set(name_to_number.values())

    for file in file_list:
        # Check if the file is a folder.
        if file["mimeType"] == "application/vnd.google-apps.folder":
            download_and_rename_files(
                drive,
                file["id"],
                os.path.join(local_path, file["title"]),
                processed_files,
                processed_files_path,
                name_to_number_path,
            )
        else:
            if file["title"] == "readme.txt":
                continue
            if file["title"] not in processed_files:
                file_path = os.path.join(local_path, file["title"])
                print(f"Downloading file {file_path}")
                file.GetContentFile(file_path)

                # Determine the new filename
                match = re.match(r"(\d+)_pattern_", file["title"])

                if match:  # Already correct filename
                    number = int(match.group(1))
                    if number not in used_numbers:
                        used_numbers.add(number)
                    else:
                        number = generate_unique_number(
                            match.group(0), name_to_number, used_numbers
                        )

                else:
                    name_match = re.match(
                        r"([^\W_]+)(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s(\w+)\s(\d+)",
                        file["title"],
                    )
                    if name_match:
                        name = name_match.group(1)
                        day = name_match.group(2)
                        month = name_match.group(3)
                        day_num = name_match.group(4)
                        name_day_month_daynum = f"{name}_{day}_{month}_{day_num}"
                        number = generate_unique_number(
                            name_day_month_daynum, name_to_number, used_numbers
                        )

                    else:
                        print(
                            f"Error: Filename {file['title']} does not match expected patterns."
                        )
                        continue
                new_filename = f"{number}_pattern_{str(uuid.uuid4())}.json"
                new_filepath = os.path.join(local_path, new_filename)
                if number < past_number:
                    print(
                        f"Watch out, there can be a mistake when processing files: Original name {file['title']},"
                        f" changed_name: {new_filename}"
                    )
                past_number = number

                with open(file_path, "r") as f:
                    d = json.load(f)
                with open(new_filepath, "w") as f:
                    json.dump(d, f)
                os.remove(file_path)  # Remove the original file
                add_to_processed_files(processed_files_path, file["title"])
                print(f"File renamed to: {new_filename}")
            else:
                print(f'File {file["title"]} already processed, skipping download.')
    # Save the updated mapping of names to numbers


if __name__ == "__main__":
    drive = authenticate()

    # ID of the main folder you want to download files from.
    main_folder_id = "1y2VBNGt5vQzsxQsWKjFaukaUaOrY4azH"

    # Local path to store the downloaded folder structure.
    local_main_folder_path = "data"
    processed_files_path = "processed_files.txt"
    name_to_number_path = "name_to_number.json"
    processed_files = read_processed_files(processed_files_path)
    used_numbers = set()
    download_and_rename_files(
        drive,
        main_folder_id,
        local_main_folder_path,
        processed_files,
        processed_files_path,
        name_to_number_path,
    )

    print("All files have been downloaded.")
