import os
import argparse
import logging
from s3_utils import upload_model_to_s3

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Upload model to S3')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--s3-key', default='models/model.pth', help='S3 key for the model')
    parser.add_argument('--local-path', default='./models/model.pth', 
                       help='Local path to the model file')
    
    args = parser.parse_args()
    
    logger.info(f"Uploading {args.local_path} to s3://{args.bucket}/{args.s3_key}")
    try:
        success = upload_model_to_s3(args.local_path, args.bucket, args.s3_key)
        if success:
            logger.info("Upload successful!")
        else:
            logger.error("Upload failed.")
    except Exception as e:
        logger.error(f"Error during upload: {str(e)}")

if __name__ == "__main__":
    main()
