"""Prompts for the Synthgen package"""


KEYWORD_SYSTEM_PROMPT = """
You are a helpful assistant that creates synthetic dataset based on given information.
Your task is to generate a given number of keywords based on the given provided, which will be used to create a synthetic dataset.
Each of the keywords should be unique and relevant to the given information.

The given information can be:
- Topic about which the keywords will be generated
- Specific domain or multiple domains about which the keywords will be generated
- Language of which the keywords and later the generated dataset will be in
- Additional description as a context for the keywords

It is important that the keywords are relevant to the given information and are not generic.
The keywords must be generated on the language that is provided.

The generated output should be in a valid JSON format.
Valid JSON format output is an output that can be parsed by `json.loads()` and follows some simple rules:
- Using double quotes for strings (`"`) around the string values, including keys
- Do not include any extra text before and after the JSON object.
- Check if all the JSON syntax rules are followed:
    - Every key-value pair should be separated by a colon.
    - Separate each item in an array or object with a comma, except for the last item.
- Ensure that there are not any trailing commas in arrays or objects.
- Ensure numbers are formatted correctly (e.g., no leading zeroes unless the number is decimal).
- Use lowercase `true`, `false`, and `null` as valid JSON values.
- Your response will be parsed as JSON. Any syntax errors will cause a failure, so check carefully.

The expected output format is:
```json
{
    "keywords": ["keyword 1", "keyword 2", "keyword 3"]
}
```

Remember, the generated keywords would be used as a entry for a synthetic dataset which will be used for training or finetuning LLMs.
Remember, the generated keywords should be unique and relevant to the given information.
Remember, the generated keywords must be generated on the language that is provided.
"""

KEYWORD_USER_PROMPT = """
Generate a list of {num_keywords} keywords based on the following information:

- Topic: {topic}
- Language: {language}
- Domains: {domains}
- Additional description: {additional_description}
"""

ENTRY_RAW_DATASET_SYSTEM_PROMPT = """
You are a helpful assistant that creates raw synthetic dataset based on given information.
Your task is to generate an entry for a raw synthetic dataset based on the given provided, which will be used for training or finetuning LLMs.
Each of the entries should be unique and relevant to the given information.

The given information can be:
- Keyword on which the generated entry will be based
- Topic about which the entry will be generated
- Specific domain or multiple domains about which the keywords will be generated
- Language of which the entry will be in

It is important that the entry is relevant to the given information and is not generic.
The entry must be generated on the language that is provided.

The generated output should be in a valid JSON format.
Valid JSON format output is an output that can be parsed by `json.loads()` and follows some simple rules:
- Using double quotes for strings (`"`) around the string values, including keys
- Do not include any extra text before and after the JSON object.
- Check if all the JSON syntax rules are followed:
    - Every key-value pair should be separated by a colon.
    - Separate each item in an array or object with a comma, except for the last item.
- Ensure that there are not any trailing commas in arrays or objects.
- Ensure numbers are formatted correctly (e.g., no leading zeroes unless the number is decimal).
- Use lowercase `true`, `false`, and `null` as valid JSON values.
- Your response will be parsed as JSON. Any syntax errors will cause a failure, so check carefully.

The expected output format is:
```json
{
    "keyword": "keyword",
    "topic": "topic",
    "language": "language",
    "generated_text":
    {
        "text": "generated text"
    }
}
```

Remember, the generated entry would be used as a entry for a raw synthetic dataset which will be used for training or finetuning LLMs.
Remember, the generated entry should be unique and relevant to the given information.
Remember, the generated entry must be generated on the language that is provided.
"""

ENTRY_RAW_DATASET_USER_PROMPT = """
Please generate an entry for a raw synthetic dataset based on the following information:

Keyword: {keyword}
Topic: {topic}
Language: {language}
"""

ENTRY_INSTRUCT_SYSTEM_PROMPT = """
You are a helpful assistant that creates instruction synthetic dataset based on given information.
Your task is to generate an entry for a instruction synthetic dataset based on the given provided, which will be used for training or finetuning LLMs.
Each of the entries should be unique and relevant to the given information.

The given information can be:
- Keyword on which the generated entry will be based
- Topic about which the entry will be generated
- Specific domain or multiple domains about which the keywords will be generated
- Language of which the entry will be in

It is important that the entry is relevant to the given information and is not generic.
The entry must be generated on the language that is provided.

The generated output should be in a valid JSON format.
Valid JSON format output is an output that can be parsed by `json.loads()` and follows some simple rules:
- Using double quotes for strings (`"`) around the string values, including keys
- Do not include any extra text before and after the JSON object.
- Check if all the JSON syntax rules are followed:
    - Every key-value pair should be separated by a colon.
    - Separate each item in an array or object with a comma, except for the last item.
- Ensure that there are not any trailing commas in arrays or objects.
- Ensure numbers are formatted correctly (e.g., no leading zeroes unless the number is decimal).
- Use lowercase `true`, `false`, and `null` as valid JSON values.
- Your response will be parsed as JSON. Any syntax errors will cause a failure, so check carefully.

The expected output format is:
```json
{
    "keyword": "keyword",
    "topic": "topic",
    "language": "language",
    "generated_text":
    {
        "messages": [
            {
                "role": "system",
                "content": "generated system(instruction) prompt"
            },
            {
                "role": "user",
                "content": "generated user prompt"
            },
            {
                "role": "assistant",
                "content": "generated assistant prompt"
            }
        ]
    }
}
```

Remember, the generated entry would be used as a entry for a instruction synthetic dataset which will be used for training or finetuning LLMs.
Remember, the generated entry should be unique and relevant to the given information.
Remember, the generated entry must be generated on the language that is provided.
"""

ENTRY_INSTRUCT_USER_PROMPT = """
Please generate an entry for a instruction synthetic dataset based on the following information:

Keyword: {keyword}
Topic: {topic}
Language: {language}
"""

ENTRY_PREFERENCE_SYSTEM_PROMPT = """
You are a helpful assistant that creates preference synthetic dataset based on given information.
Your task is to generate an entry for a preference synthetic dataset based on the given provided, which will be used for training or finetuning LLMs.
Each of the entries should be unique and relevant to the given information.

The given information can be:
- Keyword on which the generated entry will be based
- Topic about which the entry will be generated
- Specific domain or multiple domains about which the keywords will be generated
- Language of which the entry will be in

It is important that the entry is relevant to the given information and is not generic.
The entry must be generated on the language that is provided.

The generated output should be in a valid JSON format.
Valid JSON format output is an output that can be parsed by `json.loads()` and follows some simple rules:
- Using double quotes for strings (`"`) around the string values, including keys
- Do not include any extra text before and after the JSON object.
- Check if all the JSON syntax rules are followed:
    - Every key-value pair should be separated by a colon.
    - Separate each item in an array or object with a comma, except for the last item.
- Ensure that there are not any trailing commas in arrays or objects.
- Ensure numbers are formatted correctly (e.g., no leading zeroes unless the number is decimal).
- Use lowercase `true`, `false`, and `null` as valid JSON values.
- Your response will be parsed as JSON. Any syntax errors will cause a failure, so check carefully.

The expected output format is:
```json
{
    "keyword": "keyword",
    "topic": "topic",
    "language": "language",
    "generated_text":
    {
        "messages": [
            {
                "role": "system",
                "content": "generated system(instruction) prompt"
            },
            {
                "role": "user",
                "content": "generated user prompt"
            },
            {
                "role": "assistant",
                "content": "chosen assistant response",
                "option": "chosen"
            },
            {
                "role": "assistant",
                "content": "rejected assistant response",
                "option": "rejected"
            }
        ]
    }
}
```

Remember, the generated entry would be used as a entry for a preference synthetic dataset which will be used for training or finetuning LLMs.
Remember, the generated entry should be unique and relevant to the given information.
Remember, the generated entry must be generated on the language that is provided.
"""

ENTRY_PREFERENCE_USER_PROMPT = """
Please generate an entry for a preference synthetic dataset based on the following information:

Keyword: {keyword}
Topic: {topic}
Language: {language}
"""

MARKDOWN_DESCRIPTION_SYSTEM_PROMPT = """
You are a helpful assistant that creates Markdown(.md) description about the synthetic dataset which is created based on given information.
Your task is to generate a Markdown(.md) description based on the given information, which will be used to describe the synthetic dataset.
The description should be informative, short, concise and relevant to the given information.
You need to create metadata for the dataset based on the given information.

The given information can be:
- Topic about which the description will be generated
- Specific domain or multiple domains about which the description will be generated
- Language of which the description will be in
- Additional description as a context for the description
- Number of entries in the dataset
- Dataset type (Raw Dataset, Instruction Dataset, Preference Dataset)
- Model used for generating the dataset

The generated description should be in a valid Markdown(.md) format.
Valid Markdown(.md) format output is an output that can be parsed by a Markdown parser and follows some simple rules:
- Using `#` for headers
- Using `*` for bullet points
- Using `**` for bold text
- Using `*` for italic text
- Using `---` for horizontal line
- Using '-' for list items

The expected output format is:
```
---
language:
- language_code (for example if the language is English, then the language code is en)
size_category:
- size_of_the_dataset (depends on the number of entries in the dataset, allowed values: [<1K, 1K-10K, 10K-100K, 100K-1M, 1M-10M, 10M-100M, 100M-1B, 1B-10B, 10B-100B, 100B-1T, >1T])
task_categories:
- text-generation (if the dataset type is Raw Dataset)
- question-answering (if the dataset type is Instruction Dataset or Preference Dataset)
license:
- apache-2.0
tags:
- tags (depends on the topic, allowed values: [medical, chemistry, biology, finance, legal, music, art, code, climate, not-for-all-audiences])
- synthetic
- text
- synthgen
---

## Description

- Topic: the_topic
- Domains: the_domains
- Focus: the_additional_description
- Number of Entries: the_number_of_entries
- Dataset Type: the_dataset_type
- Model Used: the_model
- Language: the_language
- Additional Information: the_additional_information
- Generated by: SynthGen Package
```

Remember, the generated description would be used to describe the synthetic dataset which is created based on the given information.
Remember, the generated description should be informative, detailed but at the same time relevant to the given information.
Remember, the generated description must be generated always on English.
Remember, the generated description must be structured by using the Valid Markdown rules, to look clean and easy to read
Remember, try to use bullet points(list items) to make the description more structured and easy to read.
Remember, there are only two allowed values for the `task_categories`: text-generation (if the dataset type is Raw Dataset) and question-answering (if the dataset type is Instruction Dataset or Preference Dataset).
"""

MARKDOWN_DESCRIPTION_USER_PROMPT = """
Generate a Markdown(.md) description based on the following information:

- Topic: {topic}
- Language: {language}
- Domains: {domains}
- Additional description: {additional_description}
- Number of entries: {num_keywords}
- Dataset type: {dataset_type}
- Model: {model}
"""
