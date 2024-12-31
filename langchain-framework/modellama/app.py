import requests
import json
import gradio as gr

url = "http://localhost:11434/api/generate"
headers={
    'Content-Type': 'application/json'
}
history=[]

def generate_response(prompt):
    history.append(prompt)
    final_history="\n".join(history)
    
    data={
        "model":"deepakmodel",
        "prompt": final_history,
        "stream": False
    }
    
    response=requests.post(url, headers=history, data=json.dumps(data))
    
    if response.status_code==200:
        response=response.text
        data=json.load(response)
        actual_response=data['response']
        return actual_response
    else:
        print(f"Error", response.text)
        
interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=4, placeholder="Enter your prompt"),
    outputs="text"
)
interface.launch(share=True)