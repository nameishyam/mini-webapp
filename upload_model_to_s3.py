import os
import argparse
from s3_utils import upload_model_to_s3

def main():
    parser = argparse.ArgumentParser(description='Upload model to S3')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--s3-key', default='models/model.pth', help='S3 key for the model')
    parser.add_argument('--local-path', default='./models/model.pth', 
                       help='Local path to the model file')
    
    args = parser.parse_args()
    
    print(f"Uploading {args.local_path} to s3://{args.bucket}/{args.s3_key}")
    success = upload_model_to_s3(args.local_path, args.bucket, args.s3_key)
    
    if success:
        print("Upload successful!")
    else:
        print("Upload failed.")

if __name__ == "__main__":
    main()
