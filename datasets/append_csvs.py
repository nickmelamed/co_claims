# append_csv.py
import boto3
import pandas as pd
from io import StringIO

s3 = boto3.client('s3')
MASTER_KEY = 'combined/master.csv'

def lambda_handler(event, context):
    record = event['Records'][0]['s3']
    bucket = record['bucket']['name']
    new_key = record['object']['key']

    # Ignore the master file itself
    if new_key == MASTER_KEY:
        return {'statusCode': 200, 'body': 'Skipped master file'}

    # Read the newly uploaded CSV
    new_obj = s3.get_object(Bucket=bucket, Key=new_key)
    new_df = pd.read_csv(new_obj['Body'])

    # Read master CSV (or start fresh)
    try:
        master_obj = s3.get_object(Bucket=bucket, Key=MASTER_KEY)
        master_df = pd.read_csv(master_obj['Body'])
        master_df = pd.concat([master_df, new_df], ignore_index=True)
    except s3.exceptions.NoSuchKey:
        master_df = new_df

    # Write back
    buf = StringIO()
    master_df.to_csv(buf, index=False)
    s3.put_object(Bucket=bucket, Key=MASTER_KEY, Body=buf.getvalue())

    print(f"Appended {new_key} → {len(new_df)} rows added, master now {len(master_df)} rows")
    return {'statusCode': 200}