import requests
import json

url = "http://localhost:11434/api/generate"

data = {

    "model": "llama3.2",
    "prompt": "Who is Cristiano Ronaldo",

}

response = requests.post(url, json=data, stream=True)

#Check the response status

if response.status_code== 200:
    print("Generated Text:", end=" ", flush=True)
#iterate over the streaming process

for line in response.iter_lines():
    if line:
        #Decode the line and parse the json
        decode_line = line.decode("utf-8")
        result = json.loads(decode_line)

        #get text from response
        generated_text = result.get("response", "")
        print(generated_text, end="", flush= True)

    else:
        print("Error:", response.status_code, response.text)