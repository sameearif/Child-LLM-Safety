import os
import json
import argparse
import random
from tqdm import tqdm
from openai import OpenAI
from utils import load_dataset

def add_age(dataset):
    new_dataset = {}
    for category in dataset:
        new_dataset[category] = []
        for prompt in dataset[category]:
            age = random.randint(8, 15)
            new_dataset[category].append(f'Age: {age}\n\n{prompt}')
    return new_dataset

def generate_resppnses(client, model_name, dataset):
    responses = {}
    for category in dataset:
        responses[category] = []
        for prompt in tqdm(dataset[category]):
            response = client.chat.completions.create(
            model=model_name,
            messages=[
                    {'role': 'user', 'content': prompt}
                ]
            )
            response = response.choices[0].message.content
            responses[category].append({'prompt': prompt, 'response': response})
            with open(f'responses/{model_name.split("/")[-1]}.json', 'w', encoding='utf-8') as f:
                json.dump(responses, f, indent='\t', ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=['meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'], required=True, help='Model name, e.g., gpt-5 or llama-3')
    parser.add_argument('--dataset_name', type=str, choices=["CSB.json", "CSB_Cues.json"], required=True, help='Dataset file name')
    parser.add_argument('--add_age', type=bool, help='Explicitly add age in the prompt')
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)
    if args.add_age:
        dataset = add_age(dataset)

    client = OpenAI(
        api_key=os.getenv('DEEPINFRA_API_KEY'),
        base_url='https://api.deepinfra.com/v1/openai',
    )
    generate_resppnses(client, args.model_name, dataset)

if __name__ == "__main__":
    main()