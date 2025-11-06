import json

def load_dataset(file_name):
    dataset = json.load(open(f'datasets/{file_name}'))
    return dataset

def load_responses(model_name):
    responses = json.load(open(f'responses/{model_name}.json'))
    return responses
