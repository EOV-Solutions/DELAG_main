import json

def read_geojson_file(filepath: str):
    """
    Reads data from a .geojson file and returns it as a Python dictionary.

    Args:
        filepath (str): The path to the .geojson file.

    Returns:
        dict: The content of the .geojson file as a dictionary.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        return geojson_data
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file at {filepath}. "
              "Please ensure it's a valid GeoJSON file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Example usage:
# If you have a file named 'your_data.geojson' in the same directory:
# geojson_content = read_geojson_file('your_data.geojson')

if __name__ == "__main__":
    # Or specify a full path:
    geojson_content = read_geojson_file('/mnt/hdd12tb/code/nhatvm/DELAG_main/Grid_50K_MatchedDates.geojson')

    print(geojson_content['features'][1])
    # if geojson_content:
    #     print("Successfully read GeoJSON data.")
    #     print(f"Type of GeoJSON object: {geojson_content.get('type')}")
    #     if 'features' in geojson_content:
    #         print(f"Number of features: {len(geojson_content['features'])}")
    print()
    print()
    print(geojson_content['features'][2])
