import openeo
from openeo.processes import filter_temporal, filter_bbox

if __name__ == "__main__":
    # Connect and authenticate
    connection = openeo.connect("https://openeo.dataspace.copernicus.eu")
    connection.authenticate_oidc()

    # Define AOI and time range
    bbox = {"west": 4.0, "east": 5.0, "south": 50.0, "north": 51.0}
    time_range = ("2023-01-01", "2023-01-31")

    # Load the SLSTR LST collection
    datacube = connection.load_collection(
        collection_id="SENTINEL3_SLSTR_L2_LST",
        bands=["LST"],
        temporal_extent=time_range,
        spatial_extent=bbox
    )

    # Create and start batch job
    job = datacube.create_job(out_format="JSON")
    job_id = job.job_id
    print(f"Job ID: {job_id}")

    # Monitor job status with timeout
    import time
    timeout = 600  # 10 minutes
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = job.status()
        print(f"Job '{job_id}': {status}")
        if status in ["finished", "error", "canceled"]:
            break
        time.sleep(10)

    # Retrieve results or logs if finished or failed
    if status == "finished":
        results = job.get_results()
        metadata = results.get_metadata()
        print("Available timestamps:")
        for item in metadata.get("features", []):
            print(item["properties"]["datetime"])
    elif status == "error":
        print("Job failed. Retrieving logs...")
        print(job.logs())
    else:
        print(f"Job did not complete within {timeout} seconds. Current status: {status}")