import boto3
import json

prompt_data="""
Act as shakespeare and write a poem on machine learning.
"""

client = boto3.client('bedrock-runtime')

payload={
    "prompt": prompt_data,
    "max_gen_len": 512,
    "temperature":0.5,
    "top_p": 0.9
}

body=json.dumps(payload)

response = client.invoke_model(
    body=body,
    contentType='application/json',
    accept='application/json',
    modelId='meta.llama3-3-70b-instruct-v1:0'
)
print(response)
