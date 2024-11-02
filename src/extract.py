import os
import pandas as pd


def extract_affinities_from_file(file_path: str, software: str = "vina") -> dict:
    """
    Extracts docking results from the specified software output file.

    Parses a file from molecular docking software to extract affinity and,
    for gnina, additional CNN-based scores. The results are returned in a
    dictionary containing lists of these results, tailored to the specified software.

    Parameters:
    - file_path (str): The path to the docking results file.
    - software (str): Identifier for the docking software used, affecting the data parsed.
                      Supported values are 'vina', 'smina', 'qvina', 'vina-gpu-2', and 'gnina'.

    Returns:
    - dict: A dictionary with keys for 'affinity', and depending on the software,
            'cnn_pose_score' and 'cnn_affinity'. Only 'gnina' includes all three.
            Each key maps to a list of float values extracted from the file.

            Example:
                {
                    'affinity': [float, ...],
                    'cnn_pose_score': [float, ...],  # Only if software is 'gnina'
                    'cnn_affinity': [float, ...]    # Only if software is 'gnina'
                }
    """
    results = {"affinity": [], "cnn_pose_score": [], "cnn_affinity": []}

    # Open the file and process line by line
    with open(file_path, "r") as file:
        for line in file:
            clean_line = line.strip()
            # Identify the start of the table containing the data
            if clean_line.startswith("mode |"):
                # Skip the next two lines which are headers for the columns
                next(file)  # Skip column titles line
                next(file)  # Skip column units line
                # Process each line of data
                for data_line in file:
                    data_parts = data_line.split()
                    # Check if the line is still part of the table
                    if len(data_parts) < 2 or not data_parts[0].isdigit():
                        break  # End of data table
                    try:
                        # Append data based on the software type
                        if software == "gnina":
                            results["affinity"].append(float(data_parts[1]))
                            results["cnn_pose_score"].append(float(data_parts[2]))
                            results["cnn_affinity"].append(float(data_parts[3]))
                        else:
                            results["affinity"].append(float(data_parts[1]))
                    except ValueError:
                        continue  # Handle conversion error and move to the next line

    # Remove keys for data not applicable to software other than gnina
    if software != "gnina":
        del results["cnn_pose_score"]
        del results["cnn_affinity"]

    return results


def extract_all_affinities(directory: str, software: str, best: bool = False) -> None:
    """
    Processes all .txt files in a specified directory for a given docking software, extracting
    docking results and compiles them into a single CSV file. The filenames (without .txt)
    are used as column headers (ID_1, ID_2, etc.), and each column contains either a list of
    affinity values or just the best affinity value from each file, based on the `best` parameter.

    Parameters:
    - directory (str): The directory containing .txt files to process.
    - software (str): The docking software type of the files to process.
    - best (bool): If True, only the best affinity (the first one) is saved.
                   If False, all affinities are saved in a list.

    Returns:
    - None: This function saves the output directly to a CSV file named 'records.csv'.
    """

    data = []

    # Iterate through each file in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            # Use filename without extension as the ID
            file_id = filename[:-4]
            # Extract the data using the previously defined function
            extracted_data = extract_affinities_from_file(file_path, software)

            # Depending on the 'best' parameter, append either all affinities or just the best one
            if best and extracted_data["affinity"]:
                # Only the best affinity is appended
                affinity_value = extracted_data["affinity"][0]
            else:
                # All affinities are appended as a list
                affinity_value = extracted_data["affinity"]

            data.append({"ID": file_id, f"{software}": affinity_value})
    return data


def compile_multiple_software_results(
    software_list: list, base_directory: str, best: bool = False
) -> pd.DataFrame:
    """
    Compiles affinity results from multiple docking software outputs, stored in separate directories,
    into a single DataFrame.

    Parameters:
    - software_list (list): List of software names, each corresponding to a subdirectory.
    - base_directory (str): Base directory path containing software subdirectories.
    - best (bool): If True, only the best affinity (the first one) is extracted from each file.

    Returns:
    - pd.DataFrame: A DataFrame where each row represents an ID and columns include
                    affinities for each software type, labeled as `{software}_affinity`.
    """
    compiled_data = pd.DataFrame()

    # Iterate through each software type
    for software in software_list:
        directory = os.path.join(base_directory, software, "Log")
        if not os.path.isdir(directory):
            print(f"No directory found for {software}: {directory}")
            continue  # Skip this software if the directory is not found

        # Call the function to process all files in the directory
        software_results = extract_all_affinities(directory, software, best)

        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(software_results)

        # Rename affinity columns to include the software name
        df.rename(columns={software: f"{software}_affinity"}, inplace=True)

        # Merge this software's results into the compiled DataFrame
        if compiled_data.empty:
            compiled_data = df
        else:
            compiled_data = pd.merge(compiled_data, df, on="ID", how="outer")

    return compiled_data
