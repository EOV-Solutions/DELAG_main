
# ETL Processing Instructions

Use this command to process the data with the ETL module:

```bash
python -m ETL_data_retrieval_module.main \
    --roi_name "your_roi_name" \
    --task_ids "etl_task_mapping.json" \
    --datasets lst \
    --output_folder "./processed_data"
```

# Manual Server Setup

If you want to set up a simple server for the ZIP files:

1. Start the upload server:
```bash
python simple_upload_server.py --port 8000 --storage_dir ./server_storage
```

2. Copy ZIP files to server storage:

cp ./gee_etl_ready/zips/l8l2_5.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_d.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_f.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_5.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_2.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_e.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_9.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_9.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_-.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_3.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_4.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_3.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_1.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_-.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_4.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_9.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_0.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_8.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_-.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_8.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_5.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_9.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_c.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_-.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_1.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_c.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_0.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_5.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_a.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_5.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_5.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_1.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_1.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_c.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_2.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l8l2_8.zip ./server_storage/l8l2/

cp ./gee_etl_ready/zips/l9l2_6.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_d.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_d.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_7.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_1.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_0.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_c.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_5.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_-.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_f.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_0.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_d.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_0.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_-.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_4.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_b.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_7.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_8.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_-.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_b.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_6.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_a.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_e.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_-.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_c.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_9.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_c.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_1.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_6.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_a.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_2.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_7.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_e.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_b.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_a.zip ./server_storage/l9l2/

cp ./gee_etl_ready/zips/l9l2_5.zip ./server_storage/l9l2/
