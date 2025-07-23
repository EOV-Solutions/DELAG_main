import WaPOR
import datetime
import os

# Set your WaPOR API token
API_TOKEN = "your_api_token_here"  # Replace with your actual token

# Define the region in Vietnam (example: Mekong Delta region)
bbox = {
    "min_lon": 104.5,  # Minimum longitude
    "max_lon": 106.5,  # Maximum longitude
    "min_lat": 8.5,    # Minimum latitude
    "max_lat": 10.5    # Maximum latitude
}

# Define dataset parameters
dataset = "L1_AETI_D"  # Level 1 Actual Evapotranspiration and Interception, dekadal
start_date = datetime.date(2023, 1, 1)
end_date = datetime.date(2024, 12, 31)
output_dir = "wapor_data_vietnam"  # Output directory

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Authenticate with WaPOR API
WaPOR.API.set_access_token(API_TOKEN)

# Download data
try:
    WaPOR.download_data(
        dataset=dataset,
        start_date=start_date,
        end_date=end_date,
        bbox=bbox,
        output_dir=output_dir,
        time_step="DEKADAL"  # Options: DAILY, DEKADAL, MONTHLY, YEARLY
    )
    print(f"Data downloaded successfully to {output_dir}")
except Exception as e:
    print(f"Error downloading data: {e}")