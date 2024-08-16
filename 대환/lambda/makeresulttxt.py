import json
import boto3

# AWS 클라이언트 생성
sagemaker_runtime_client = boto3.client('sagemaker-runtime')
s3_client = boto3.client('s3')
rekognition_client = boto3.client('rekognition')

def lambda_handler(event, context):
    try:
        # S3 이벤트에서 버킷 이름과 파일 키 가져오기
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        # 이미지 파일인지 확인
        if not key.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"File {key} is not an image. Skipping Rekognition.")
            return {
                'statusCode': 400,
                'body': json.dumps(f"File {key} is not an image. Skipping Rekognition.")
            }
        
        # Rekognition을 사용하여 얼굴 바운딩 박스 추출
        rekognition_response = rekognition_client.detect_faces(
            Image={
                'S3Object': {
                    'Bucket': bucket,
                    'Name': key
                }
            },
            Attributes=['ALL']
        )
        
        # 얼굴 바운딩 박스 정보 추출
        face_details = rekognition_response['FaceDetails']
        
        if face_details:
            bounding_box = face_details[0]['BoundingBox']
            
            # SageMaker 엔드포인트 호출
            test_input = {
                'bucket': bucket,
                'key': key,
                'bounding_box': {
                    'Left': bounding_box['Left'],
                    'Top': bounding_box['Top'],
                    'Width': bounding_box['Width'],
                    'Height': bounding_box['Height']
                }
            }
            
            # 입력 데이터를 S3에 JSON 형식으로 저장 (비동기 추론을 위해 필요)
            input_key = f'input/{key.split("/")[-1].split(".")[0]}.json'
            input_location = f's3://{bucket}/{input_key}'
            s3_client.put_object(
                Bucket=bucket,
                Key=input_key,
                Body=json.dumps(test_input)
            )
            
            # 비동기 추론 호출
            response = sagemaker_runtime_client.invoke_endpoint_async(
                EndpointName='asy',
                InputLocation=input_location,
                ContentType='application/json',  # 올바른 Content-Type 설정
                InvocationTimeoutSeconds=3600  # 최대 1시간 설정 가능
            )
            
            output_location = response['OutputLocation']
            print("Inference request submitted. Output will be stored in:", output_location)
            
            return {
                'statusCode': 200,
                'body': json.dumps('Inference request submitted. Check S3 for results.')
            }
        else:
            print(f"No faces detected in the image for {key}")
            return {
                'statusCode': 200,
                'body': json.dumps(f"No faces detected in the image for {key}")
            }
    except Exception as e:
        print(f"Error processing {key} from bucket {bucket}. Error: {str(e)}")
        return {
            'statusCode': 500,
                'body': json.dumps(f"Error processing image: {str(e)}")
            }
