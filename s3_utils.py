import os
import boto3
from botocore.exceptions import NoCredentialsError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model_from_s3(bucket_name, s3_model_key, local_model_path):
    """
    Download model from S3 bucket if it doesn't exist locally
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
    
    # Check if model already exists locally
    if os.path.exists(local_model_path):
        logger.info(f"Model already exists at {local_model_path}")
        return True
    
    try:
        logger.info(f"Downloading model from s3://{bucket_name}/{s3_model_key} to {local_model_path}")
        
        # Initialize S3 client
        s3 = boto3.client('s3')
        
        # Download the file
        s3.download_file(bucket_name, s3_model_key, local_model_path)
        logger.info("Model downloaded successfully")
        return True
        
    except NoCredentialsError:
        logger.error("AWS credentials not available")
        return False
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False

def upload_model_to_s3(file_path, bucket_name, s3_key):
    """
    Upload model to S3 bucket
    """
    try:
        logger.info(f"Uploading {file_path} to s3://{bucket_name}/{s3_key}")
        
        # Initialize S3 client
        s3 = boto3.client('s3')
        
        # Upload the file
        s3.upload_file(file_path, bucket_name, s3_key)
        logger.info("Model uploaded successfully")
        return True
        
    except FileNotFoundError:
        logger.error(f"The file {file_path} was not found")
        return False
    except NoCredentialsError:
        logger.error("AWS credentials not available")
        return False
    except Exception as e:
        logger.error(f"Error uploading model: {str(e)}")
        return False
