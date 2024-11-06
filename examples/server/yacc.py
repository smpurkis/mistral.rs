from pathlib import Path
from openai import OpenAI

from json_schema_to_regex import generate_regex_from_schema

client = OpenAI(api_key="foobar", base_url="http://localhost:8230/v1/")

with open("examples/server/c.y", "r") as f:
    c_yacc = f.read()


schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "$id": "https://example.com/employee.schema.json",
    "title": "Record of employee",
    "description": "This document records the details of an employee",
    "type": "object",
    "properties": {
        "id": {"description": "A unique identifier for an employee", "type": "string"},
        # "name": {
        #     "description": "name of the employee",
        #     "type": "string",
        #     "minLength": 2,
        # },
        # "age": {"description": "age of the employee", "type": "number", "minimum": 16},
        # "hobbies": {
        #     "description": "hobbies of the employee",
        #     "type": "object",
        #     "properties": {
        #         "indoor": {
        #             "type": "array",
        #             "items": {"description": "List of hobbies", "type": "string"},
        #             "minItems": 1,
        #             "uniqueItems": True,
        #         },
        #         "outdoor": {
        #             "type": "array",
        #             "items": {"description": "List of hobbies", "type": "string"},
        #             "minItems": 1,
        #             "uniqueItems": True,
        #         },
        #     },
        #     "required": ["indoor", "outdoor"],
        # },
    },
    "required": ["id", "name", "age", "hobbies"],
    "additionalProperties": False,
}
# regex_grammar = r"""\{("id":-?\d+(\.\d+)?([eE][-+]?\d+)?,?"name":"(?:[^"\\\\]|\\\\.)*",?"age":-?\d+(\.\d+)?([eE][-+]?\d+)?,?"hobbies":.*?)\}"""
# regex_grammar = r"""\{("id":-?\d+(\.\d+)?([eE][-+]?\d+)?,?"name":"(?:[^"\\\\]|\\\\.)*",?"age":-?\d+(\.\d+)?([eE][-+]?\d+)?,?"hobbies":\{("indoor":\[.*?\],?"outdoor":\[.*?\])\})\}"""
regex_grammar = generate_regex_from_schema(schema)
# regex_grammar = '"(- [^\n]*\n)+(- [^\n]*)(\n\n)?"'
regex_grammar = '"(- [^\n]*\n)+(- [^\n]*)(\n\n)?"'
print(regex_grammar)
completion = client.chat.completions.create(
    model="mistral",
    messages=[
        {
            "role": "user",
            "content": "Write a document for a fake employee.",
        }
    ],
    max_tokens=256,
    frequency_penalty=1.0,
    top_p=0.1,
    temperature=0,
    extra_body={"grammar": {"type": "regex", "value": regex_grammar}},
)

print(completion.choices[0].message.content)
