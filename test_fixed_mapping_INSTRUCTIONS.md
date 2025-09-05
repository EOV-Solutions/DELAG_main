
# ETL Processing Instructions

Use this command to process the data with the ETL module:

```bash
python -m ETL_data_retrieval_module.main \
    --roi_name "your_roi_name" \
    --task_ids "test_fixed_mapping.json" \
    --datasets aster lst \
    --output_folder "./processed_data"
```

# Manual Server Setup

If you want to set up a simple server for the ZIP files:

1. Start the upload server:
```bash
python simple_upload_server.py --port 8000 --storage_dir ./server_storage
```

2. Copy ZIP files to server storage:
cp ./test_fixed_gee/zips/aster_1717ce0f-3158-4a80-b85b-910c46f1ecf4.zip ./server_storage/aster/

cp ./test_fixed_gee/zips/l8l1_0ce168b3-09fd-4fc1-8352-e87a2f39692d.zip ./server_storage/l8l1/

cp ./test_fixed_gee/zips/l8l2_d123041f-2379-40b9-8704-e951ebe82f90.zip ./server_storage/l8l2/

cp ./test_fixed_gee/zips/l9l1_a00725f2-cec7-46ea-8ed7-4991b24f72fe.zip ./server_storage/l9l1/

cp ./test_fixed_gee/zips/l9l2_8034f3ab-930d-4a6e-8e64-946e90f25e57.zip ./server_storage/l9l2/
