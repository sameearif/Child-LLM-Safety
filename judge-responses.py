import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
from utils import load_responses

def judge_responses(client, responses, system_prompt, file_name):
    evaluaton = {}
    for category in responses:
        evaluaton[category] = []
        for item in tqdm(responses[category]):
            response = client.chat.completions.create(
                model='gpt-5',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f'Question:\n{item["prompt"]}\n\n\nResponse:\n{item["response"]}'}
                ]
            )
            evaluaton[category].append({'prompt': item['prompt'], 'response': item['response'], 'evaluation': json.loads(response.choices[0].message.content)})
            with open(f'evaluation/{file_name.split("/")[-1]}.json', 'w', encoding='utf-8') as f:
                json.dump(evaluaton, f, indent='\t', ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=['Meta-Llama-3.1-8B-Instruct-Turbo'], required=True, help='Model name, e.g., gpt-5 or llama-3')
    args = parser.parse_args()
    
    responses = load_responses(args.model_name)
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    env = Environment(loader=FileSystemLoader('prompts'))
    system_prompt = env.get_template('llm-judge.jinja').render()

    judge_responses(client, responses, system_prompt, args.model_name)
    

if __name__ == "__main__":
    main()

