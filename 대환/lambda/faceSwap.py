import json
import requests
import boto3
import base64

def send_face_swap_request(source_image_base64, target_image_base64):
    # EC2 URL 입력
    url = "http://13.114.183.223:7860/reactor/image"

    # 이미지 데이터를 Base64로 인코딩하여 JSON으로 전송
    data = {
        "source_image": source_image_base64,
        "target_image": target_image_base64,
        "source_faces_index": [0],
        "face_index": [0],
        "upscaler": "None",
        "scale": 1,
        "upscale_visibility": 1,
        "face_restorer": "None",
        "restorer_visibility": 1,
        "codeformer_weight": 0.5,
        "restore_first": 1,
        "model": "inswapper_128.onnx",
        "gender_source": 0,
        "gender_target": 0,
        "save_to_file": 0,
        "result_file_path": "",
        "device": "CUDA",
        "mask_face": 0,
        "select_source": 0,
        "face_model": "None",
        "source_folder": "",
        "random_image": 0,
        "upscale_force": 0,
        "det_thresh": 0.5,
        "det_maxnum": 0
    }

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        print("Request successful")
        return response.json()  # JSON 응답을 받는다고 가정합니다.
    else:
        print(f"Request failed with status code {response.status_code}")
        print("API Error Response:", response.text)
        return None

def lambda_handler(event, context):
    # 이벤트에서 S3 버킷 이름과 파일 경로 가져오기
    s3_bucket = event['bucket']
    source_image_key = event['source_image_key']
    target_image_key = event['target_image_key']
    
    print(f"Using source image key: {source_image_key}")

    
    # S3 클라이언트 생성
    s3_client = boto3.client('s3')
    
    # S3에서 소스 이미지와 타겟 이미지 다운로드
    source_image_obj = s3_client.get_object(Bucket=s3_bucket, Key=source_image_key)
    target_image_obj = s3_client.get_object(Bucket=s3_bucket, Key=target_image_key)
    
    # 이미지 데이터를 Base64로 인코딩
    source_image_base64 = base64.b64encode(source_image_obj['Body'].read()).decode('utf-8')
    target_image_base64 = base64.b64encode(target_image_obj['Body'].read()).decode('utf-8')

    # API 요청 보내기
    result = send_face_swap_request(source_image_base64, target_image_base64)
    
    # result 내용을 로그에 출력
    print("Result from face swap API:", result)
    
    if result and 'image' in result:
        swapped_image_base64 = result['image']
        swapped_image_bytes = base64.b64decode(swapped_image_base64)
    
        # S3에 스왑된 이미지 저장
        swapped_image_key = "path/to/swapped_image.png"
        s3_client.put_object(Bucket=s3_bucket, Key=swapped_image_key, Body=swapped_image_bytes)
    
        print(f"Swapped image saved to S3 at {swapped_image_key}")
    else:
        print("No image found in result")
    
    return {
        'statusCode': 200,
        'body': 'Swapped image saved to S3 successfully.'
    }
