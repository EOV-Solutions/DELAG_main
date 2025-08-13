# DELAG_main

This project provides a pipeline for retrieving, processing, and analyzing satellite imagery data, primarily focusing on Sentinel-2 for Land Surface Temperature (LST) studies. It includes modules for data retrieval from Google Earth Engine, preprocessing of satellite images, and tools for model training and analysis.

## Features

-   Automated download and preprocessing of Sentinel-2 satellite imagery.
-   Cloud masking using Google Earth Engine's Cloud Score+.
-   Band selection and merging into single GeoTIFF files.
-   Modular structure for data retrieval, preprocessing, and modeling.
-   Scripts for hyperparameter tuning and visualization.

## Setup and Installation

### 1. Prerequisites

-   Python 3.8+
-   Git
-   A Google Earth Engine account.

### 2. Clone the Repository

Clone this repository to your local machine:

```bash
git clone <repository-url>
cd DELAG_main
```

### 3. Google Earth Engine Authentication

You need to authenticate your machine to use the Google Earth Engine API. Run the following command and follow the on-screen instructions to log in with your Google account.

```bash
earthengine authenticate
```

### 4. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 5. Install Dependencies

Install the required Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 6. Data Directory Setup

The scripts require a specific directory structure for input data (LST files). You must also set an environment variable to point to your data directory.

1.  Create your main data directory. For example: `/path/to/your/data`
2.  Inside this directory, create subfolders for each Region of Interest (ROI).
3.  Inside each ROI folder, create an `lst` subfolder containing your LST `.tif` files.

Example structure:
```
/path/to/your/data/
 |- ROI_1/
 |  |- lst/
 |  |  |- LST_file1.tif
 |  |  |- LST_file2.tif
 |- ROI_2/
 |  |- lst/
 |  |  |- LST_another_file.tif
```

4.  Set the `DELAG_DATA_DIR` environment variable to point to your main data directory.

```bash
export DELAG_DATA_DIR=/path/to/your/data
```
You can also add this line to your `~/.bashrc` or `~/.zshrc` file to make it permanent.

## Usage

The primary script for fetching and processing Sentinel-2 data is `process_s2_data.py`.

### Running the Data Processing Pipeline

With your environment activated and the `DELAG_DATA_DIR` variable set, run the main processing script:

```bash
python process_s2_data.py
```

This script will iterate through the ROI folders in your data directory, download the corresponding Sentinel-2 images from Google Earth Engine, process them, and save the results into a new `s2_images` subfolder within each ROI directory.

### Other Scripts

The repository contains other scripts for analysis and modeling:

-   `hyperparameter_tuning.py`: Perform hyperparameter tuning for models.
-   `visualize_reconstruction_comparison.py`: Visualize model results.
-   `process_lst_files.py`: Utility for processing LST files.

Run them using `python <script_name>.py`. Make sure to check the source code of each script for specific configuration or arguments they might require.

## Project Structure

-   `data/`: Default directory for data (if not using an environment variable).
-   `data_retrival_module/`: Module for retrieving data.
-   `DELAG_LST_module/`: Module related to LST processing or modeling.
-   `preprocess_module/`: Module for data preprocessing.
-   `main.py`: Main entry point for the application (currently empty).
-   `process_s2_data.py`: Main script for Sentinel-2 data processing.
-   `requirements.txt`: Project dependencies.
# DELAG_main
