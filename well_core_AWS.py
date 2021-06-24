import logging
import boto3
from botocore.exceptions import ClientError
import typing

class WellCore_S3():

    def __init__(self, session:boto3.Session, bucket_name:str="well-core"):
        self.s3 = session.resource('s3')
        self.bucket = self.s3.Bucket(bucket_name)

    def upload_file(self, file_name:str, metadata:typing.Dict, object_name:str=None) -> bool:
        """Upload a file to an S3 bucket

        :param file_name: File to upload
        :param bucket: Bucket to upload to
        :param object_name: S3 object name. If not specified then file_name is used
        :return: True if file was uploaded, else False
        """

        # If S3 object_name was not specified, use file_name
        if object_name is None:
            object_name = file_name

        # Upload the file
        try:
            response = self.bucket.upload_file(
                file_name, 
                object_name, 
                ExtraArgs={
                    "Metadata":metadata
                }
            )
        except ClientError as e:
            logging.error(e)
            return False
        return True

    def list_files(self):
        for my_bucket_object in self.bucket.objects.all():
            print(my_bucket_object)