import os
import boto3
from botocore.exceptions import NoCredentialsError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_s3_client():
    """Initialize and return an S3 client with credentials from environment"""
    # On Netlify, AWS credentials are automatically available in the environment
    # No need to explicitly provide them
    return boto3.client('s3',
                      region_name=os.getenv('AWS_DEFAULT_REGION', 'eu-north-1'))

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
        s3 = get_s3_client()
        
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
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found")
            
        logger.info(f"Uploading {file_path} to s3://{bucket_name}/{s3_key}")
        
        # Initialize S3 client
        s3 = get_s3_client()
        
        # Upload the file
        s3.upload_file(
            file_path,
            bucket_name,
            s3_key,
            ExtraArgs={
                'ACL': 'public-read',  # Adjust permissions as needed
                'ContentType': 'application/octet-stream'
            }
        )
        logger.info("Model uploaded successfully")
        return True
        
    except FileNotFoundError as e:
        logger.error(str(e))
        return False
    except NoCredentialsError:
        logger.error("AWS credentials not available. Please check your AWS credentials.")
        return False
    except Exception as e:
        logger.error(f"Error uploading model: {str(e)}")
        return False
