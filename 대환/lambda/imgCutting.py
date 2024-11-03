import json
import boto3
import numpy as np
from PIL import Image
import os

s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')
dynamodb_client = boto3.client('dynamodb')

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
        
        # 출력 UUID 추출
        output_uuid = key.split('/')[-1].replace('.out', '')

        # DynamoDB에서 RequestID 조회
        response = dynamodb_client.get_item(
            TableName='RequestMappingTable',
            Key={
                'OutputUUID': {'S': output_uuid}
            }
        )

        if 'Item' not in response:
            print("No matching RequestID found for OutputUUID:", output_uuid)
            return
        
        request_id = response['Item']['RequestID']['S']
        print(f"Extracted request_id: {request_id}")
        
        # 추출한 request_id를 사용하여 입력 JSON 파일의 키 유추
        input_json_key = f'input/{request_id}.json'
        print(f"Derived input JSON key: {input_json_key}")

        # JSON 파일 내용 읽기
        response = s3_client.get_object(Bucket=bucket, Key=input_json_key)
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
        output_key = f'path/to/{request_id}/source_image.png'
        result_image.save('/tmp/result_image.png')
        s3_client.upload_file('/tmp/result_image.png', bucket, output_key)
        print(f"Final image saved to {output_key}")
        
        # imgMake Lambda 함수 호출
        img_make_payload = {
            "bucket": bucket,
            "image_key": original_image_key,  # imgMake 함수에 필요한 데이터를 전달
            "request_id": request_id
        }

        img_make_response = lambda_client.invoke(
            FunctionName='imgMake',  # 호출할 Lambda 함수 이름
            InvocationType='RequestResponse',  # 동기 호출
            Payload=json.dumps(img_make_payload)
        )

        # imgMake Lambda 함수의 응답 처리
        img_make_response_payload = json.loads(img_make_response['Payload'].read())
        print(f"Response from imgMake Lambda: {img_make_response_payload}")

        # imgMake Lambda 함수가 성공적으로 완료된 경우에만 faceSwap Lambda 함수 호출
        if img_make_response_payload.get('statusCode') == 200:
            # faceSwap Lambda 함수 호출
            face_swap_payload = {
                "bucket": bucket,
                "request_id": request_id
            }

            face_swap_response = lambda_client.invoke(
                FunctionName='faceSwap',  # 호출할 Lambda 함수 이름
                InvocationType='RequestResponse',  # 동기 호출
                Payload=json.dumps(face_swap_payload)
            )

            # 호출된 faceSwap Lambda 함수의 응답 처리
            face_swap_response_payload = json.loads(face_swap_response['Payload'].read())
            print(f"Response from faceSwap Lambda: {face_swap_response_payload}")
        else:
            print("imgMake Lambda function failed, skipping faceSwap Lambda invocation.")
        
        # input 폴더의 JSON 파일 삭제
        s3_client.delete_object(Bucket=bucket, Key=input_json_key)
        print(f"Deleted input JSON file: {input_json_key}")
        
        # /succ 폴더 내 .out 파일 삭제하기
        s3_client.delete_object(Bucket=bucket, Key=key)
        print(f"Deleted .out file from succ folder: {key}")
        
        # /upload에 존재하는 원본 이미지 삭제하기
        s3_client.delete_object(Bucket=bucket, Key=original_image_key)
        print(f"Deleted original image from upload folder: {original_image_key}")
        
        return {
            'statusCode': 200,
            'body': json.dumps('Final image processed and saved, imgMake and faceSwap Lambda functions invoked.')
        }
    except Exception as e:
        print(f"Error processing {key} from bucket {bucket}. Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error processing image: {str(e)}")
        }
