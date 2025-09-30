#!/usr/bin/env python3
"""
AWS Rekognition Setup and Testing Script
"""

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import os
from dotenv import load_dotenv

def check_aws_credentials():
    """Check if AWS credentials are properly configured"""
    print("🔍 Checking AWS credentials...")
    
    load_dotenv(override=True)
    
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_REGION', 'us-east-1')
    
    print(f"AWS_ACCESS_KEY_ID: {'✅ Set' if access_key and access_key != 'your_actual_access_key_here' else '❌ Not set or placeholder'}")
    print(f"AWS_SECRET_ACCESS_KEY: {'✅ Set' if secret_key and secret_key != 'your_actual_secret_key_here' else '❌ Not set or placeholder'}")
    print(f"AWS_REGION: {region}")
    
    if not access_key or access_key == 'your_actual_access_key_here':
        print("\n❌ AWS credentials not properly configured!")
        print("📝 To fix this:")
        print("1. Go to AWS Console → IAM → Users → Your User → Security Credentials")
        print("2. Create new Access Key")
        print("3. Update the .env file with your actual credentials:")
        print("   AWS_ACCESS_KEY_ID=AKIA...")
        print("   AWS_SECRET_ACCESS_KEY=...")
        return False
    
    return True

def test_aws_connection():
    """Test AWS connection and permissions"""
    print("\n🔗 Testing AWS connection...")
    
    try:
        # Test basic AWS connection
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✅ AWS connection successful!")
        print(f"   Account: {identity.get('Account')}")
        print(f"   User ARN: {identity.get('Arn')}")
        
        return True
        
    except NoCredentialsError:
        print("❌ AWS credentials not found")
        return False
    except ClientError as e:
        print(f"❌ AWS connection failed: {e}")
        return False

def test_rekognition_permissions():
    """Test Rekognition service permissions"""
    print("\n👁️ Testing Rekognition permissions...")
    
    try:
        rekognition = boto3.client('rekognition')
        
        # Test listing collections (basic permission test)
        response = rekognition.list_collections()
        print(f"✅ Rekognition access successful!")
        print(f"   Existing collections: {response.get('CollectionIds', [])}")
        
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDenied':
            print("❌ Access denied to Rekognition service")
            print("📝 Required permissions:")
            print("   - rekognition:ListCollections")
            print("   - rekognition:CreateCollection")
            print("   - rekognition:DescribeCollection")
            print("   - rekognition:IndexFaces")
            print("   - rekognition:SearchFacesByImage")
            print("   - rekognition:DetectFaces")
            print("   - rekognition:DeleteFaces")
        else:
            print(f"❌ Rekognition error: {e}")
        return False

def setup_rekognition_collection():
    """Set up the Rekognition collection"""
    print("\n📦 Setting up Rekognition collection...")
    
    collection_id = os.getenv('REKOGNITION_COLLECTION_ID', 'face_collection')
    
    try:
        rekognition = boto3.client('rekognition')
        
        # Check if collection exists
        try:
            rekognition.describe_collection(CollectionId=collection_id)
            print(f"✅ Collection '{collection_id}' already exists")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                # Collection doesn't exist, create it
                print(f"📦 Creating collection '{collection_id}'...")
                rekognition.create_collection(CollectionId=collection_id)
                print(f"✅ Collection '{collection_id}' created successfully!")
                return True
            else:
                raise e
                
    except ClientError as e:
        print(f"❌ Failed to setup collection: {e}")
        return False

def test_face_detection():
    """Test face detection with a sample image"""
    print("\n🔍 Testing face detection...")
    
    try:
        import cv2
        import numpy as np
        from camera import CameraManager
        
        # Try to capture from camera
        camera = CameraManager()
        if camera.start_camera():
            print("📹 Camera available, capture a frame for testing...")
            print("Position your face and press Enter...")
            input()
            
            frame = camera.capture_frame()
            if frame is not None:
                # Convert to bytes
                _, buffer = cv2.imencode('.jpg', frame)
                image_bytes = buffer.tobytes()
                
                # Test with Rekognition
                rekognition = boto3.client('rekognition')
                response = rekognition.detect_faces(
                    Image={'Bytes': image_bytes},
                    Attributes=['ALL']
                )
                
                faces = response.get('FaceDetails', [])
                print(f"✅ Detected {len(faces)} face(s) with AWS Rekognition!")
                
                for i, face in enumerate(faces):
                    confidence = face.get('Confidence', 0)
                    print(f"   Face {i+1}: {confidence:.1f}% confidence")
                
                camera.release_camera()
                return len(faces) > 0
            else:
                print("❌ Failed to capture camera frame")
                camera.release_camera()
                return False
        else:
            print("❌ Camera not available for testing")
            return False
            
    except Exception as e:
        print(f"❌ Face detection test failed: {e}")
        return False

def main():
    """Run complete AWS setup and testing"""
    print("🚀 AWS Rekognition Setup & Testing")
    print("=" * 50)
    
    steps = [
        ("Check AWS Credentials", check_aws_credentials),
        ("Test AWS Connection", test_aws_connection),
        ("Test Rekognition Permissions", test_rekognition_permissions),
        ("Setup Rekognition Collection", setup_rekognition_collection),
        ("Test Face Detection", test_face_detection)
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        try:
            results[step_name] = step_func()
            if not results[step_name]:
                print(f"⚠️ {step_name} failed, stopping here")
                break
        except Exception as e:
            print(f"❌ {step_name} failed with exception: {e}")
            results[step_name] = False
            break
    
    print(f"\n{'='*50}")
    print("📊 SETUP SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for step_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{step_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 AWS Rekognition is ready to use!")
        print("🚀 You can now run: python web_ui.py")
        print("🔧 System will use AWS mode automatically")
    else:
        print("\n⚠️ AWS setup incomplete. Please fix the issues above.")
        print("💡 Most common issue: Invalid AWS credentials in .env file")

if __name__ == "__main__":
    main()