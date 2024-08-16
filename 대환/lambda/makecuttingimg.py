import json
import boto3
import numpy as np
from PIL import Image
import os

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    try:
        print("Lambda function started")
        
        # S3 이벤트에서 버킷 이름과 파일 키 가져오기
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        print(f"Processing file: {key} from bucket: {bucket}")
        
        # .out 파일인지 확인
        if not key.lower().endswith('.out'):
            print(f"File {key} is not an .out file. Skipping processing.")
            return {
                'statusCode': 400,
                'body': json.dumps(f"File {key} is not an .out file. Skipping processing.")
            }
        
        # input/ 폴더의 JSON 파일 목록 가져오기
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix='input/')
        json_files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.json')]
        print(f"Found JSON files: {json_files}")
        
        if not json_files:
            raise FileNotFoundError("No JSON file found in input/ folder.")
        
        # input/ 폴더에 JSON 파일이 하나만 있다고 가정하고 해당 파일 가져오기
        json_key = json_files[0]
        print(f"Using JSON file: {json_key}")
        
        # JSON 파일 내용 읽기
        response = s3_client.get_object(Bucket=bucket, Key=json_key)
        input_data = json.loads(response['Body'].read().decode('utf-8'))
        print(f"Read input data: {input_data}")
        
        # 원본 이미지 파일 경로 가져오기
        original_image_key = input_data['key']
        print(f"Original image key: {original_image_key}")
        
        # .out 파일 내용 읽기 (마스크 값 추출)
        response = s3_client.get_object(Bucket=bucket, Key=key)
        result = json.loads(response['Body'].read().decode('utf-8'))
        masks = np.array(result['masks'], dtype=np.uint8) * 255  # numpy 배열을 uint8로 변환하고 값을 0-255 범위로 조정
        print(f"Read masks: {masks.shape}")
        
        # 마스크 배열을 단일 채널로 변환 (첫 번째 채널 사용)
        mask_image = Image.fromarray(masks[0])
        
        # 원본 이미지 다운로드
        response = s3_client.get_object(Bucket=bucket, Key=original_image_key)
        original_image = Image.open(response['Body'])
        print(f"Downloaded original image: {original_image_key}")
        
        # 마스크 값 대로 이미지 자르기
        original_image.putalpha(mask_image)
        result_image = Image.new("RGBA", original_image.size)
        result_image.paste(original_image, (0, 0), mask_image)
        print("Applied mask to original image")
        
        # 최종 이미지 S3에 저장
        output_key = f'path/to/{os.path.basename(original_image_key)}'
        result_image.save('/tmp/result_image.png')
        s3_client.upload_file('/tmp/result_image.png', bucket, output_key)
        print(f"Final image saved to {output_key}")
        
        # input 폴더의 JSON 파일 삭제
        s3_client.delete_object(Bucket=bucket, Key=json_key)
        print(f"Deleted input JSON file: {json_key}")
        
        return {
            'statusCode': 200,
            'body': json.dumps('Final image processed and saved, input JSON file deleted.')
        }
    except Exception as e:
        print(f"Error processing {key} from bucket {bucket}. Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error processing image: {str(e)}")
        }
