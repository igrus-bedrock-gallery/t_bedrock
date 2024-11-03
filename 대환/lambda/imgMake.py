import base64
import boto3
import json
import os
import io
from PIL import Image
from botocore.exceptions import ClientError
import random

# AWS 서비스 클라이언트 설정
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
s3 = boto3.client('s3')

# Claude 모델 ID
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# 사용자 정의 예외 클래스
class ImageError(Exception):
    def __init__(self, message):
        self.message = message

def call_claude_haiku(base64_string, name, hope):
    prompt = f"""이미지 속 인물은 {hope}입니다. 이미지를 분석하고 가상의 인생 스토리를 만들어주세요.
        1. 특정 개인을 식별하지 마세요. 주인공은 가상의 인물이어야 합니다.
        2. 이야기는 "당신은..."으로 시작하세요.
        3. 이야기의 길이는 150자 이내로 제한하세요.
        4. 이미지 속 인물 {name}의 직업에 대한 재미있는 일화를 만들어주세요.
        5. 마지막에 '*'를 쓰고 '헤어스타일', '성별', '피부색'을 상세히 적어주세요.
        6. 인물의 어린 시절 이야기는 포함하지 마세요.
        7. 반드시 영어로 출력하세요.
        """

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_string,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    body = json.dumps(prompt_config)

    response = bedrock.invoke_model(
        body=body, modelId=MODEL_ID, accept="application/json", contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("content")[0].get("text")
    return results

def generate_text_from_image(image_file, name="신준혁", hope="소방관"):
    base64_image = base64.b64encode(image_file).decode('utf-8')
    print("Base64 Image Length:", len(base64_image))
    
    text_result = call_claude_haiku(base64_image, name, hope)
    return text_result

def generate_image(body):
    model_id = 'amazon.titan-image-generator-v2:0'
    accept = "application/json"
    content_type = "application/json"

    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=model_id,
            accept=accept,
            contentType=content_type
        )

        response_body = json.loads(response.get("body").read())
        base64_image = response_body.get("images")[0]
        image_bytes = base64.b64decode(base64_image.encode('ascii'))

        if response_body.get("error"):
            raise ImageError(f"Image generation error. Error is {response_body.get('error')}")

        return image_bytes

    except ClientError as err:
        print(f"ClientError: {err.response['Error']['Message']}")
        raise

def lambda_handler(event, context):
    try:
        # Lambda 함수가 imgCutting 함수에서 호출되었을 때의 이벤트 데이터 처리
        bucket_name = event['bucket']
        object_key = event['image_key']  # imgCutting 함수에서 전달된 이미지 키 사용
        request_id = event['request_id']

        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        image_data = response['Body'].read()

        generated_text = generate_text_from_image(image_data)
        generated_text_parts = generated_text.split("*")
        hairstyle = generated_text_parts[1] if len(generated_text_parts) > 1 else "Unknown"
        print(f"Extracted hairstyle: {hairstyle}")

        future_dream = "firefighter"
        random_seed = random.randint(0, 214783647)
        prompt = f"""
            A characteristic of person is {hairstyle}.
            A person dedicatedly working on their {future_dream}.
            A background that matches their {future_dream},
            with a detailed background.
            Looking at viewer.
            Allowing the background to be an important element of the composition.
            A Whole Face.
        """

        body = json.dumps({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 1024,
                "width": 1024,
                "cfgScale": 9.0,
                "seed": random_seed
            }
        })

        image_bytes = generate_image(body=body)
        image = Image.open(io.BytesIO(image_bytes))

        # 처리된 이미지를 S3에 저장
        destination_key = f'path/to/{request_id}/target.png'  # 이미지 이름을 명확하게 target.png로 설정
        upload_image_to_s3(image, bucket_name, destination_key)

        print(f"Image successfully processed and uploaded to {destination_key}")
        return {
            'statusCode': 200,
            'body': json.dumps('Image processed and uploaded successfully')
        }

    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error processing image: {e}")
        }

def upload_image_to_s3(image, bucket_name, file_key):
    # 이미지를 바이트 스트림으로 변환
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    # S3에 이미지 업로드
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=image_bytes, ContentType='image/png')
