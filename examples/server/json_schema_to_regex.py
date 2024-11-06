import re
import json


def handle_enum(details: dict) -> str:
    return f'"({"|".join(map(re.escape, details["enum"]))})"'


def handle_string(details: dict) -> str:
    # return '"(?:[^"!\\\\\\\\]|\\\\\\\\.)*"'
    # simple
    # return '"(?:[^"]|.)*"'
    return '""'


def handle_number(details: dict) -> str:
    return "-?\\d+(\\.\\d+)?([eE][-+]?\\d+)?"


def handle_integer(details: dict) -> str:
    return "-?\\d+"


def handle_boolean(details: dict) -> str:
    return "true|false"


def handle_array(details: dict) -> str:
    # assumes that the array has a single type
    type_pattern = handle_type({"type": details["items"].get("type", "string")})
    # arrays_pattern = f"({type_pattern},)*{type_pattern}"
    arrays_pattern = f"({type_pattern})*{type_pattern}"

    return f"\\[{arrays_pattern}\\]"


def handle_type(details: dict) -> str:
    if "enum" in details:
        pattern = handle_enum(details)
    elif details["type"] == "string":
        pattern = handle_string(details)
    elif details["type"] == "number":
        pattern = handle_number(details)
    elif details["type"] == "integer":
        pattern = handle_integer(details)
    elif details["type"] == "boolean":
        pattern = handle_boolean(details)
    elif details["type"] == "object":
        # Recursively generate regex for nested objects
        pattern = generate_regex_from_schema(details)
    elif details["type"] == "array":
        # pattern = handle_array(details)
        pattern = "\\[.*?\\]"
        # pattern = ""

    return pattern


def generate_regex_from_schema(schema: dict) -> str:
    if schema["type"] != "object":
        raise ValueError("This converter only supports object schemas")

    properties: dict = schema.get("properties", {})
    required = schema.get("required", [])

    property_patterns = []
    for prop, details in properties.items():
        prop_pattern = f'"{prop}":'
        prop_pattern += handle_type(details)

        if prop in required:
            property_patterns.append(prop_pattern)
        else:
            property_patterns.append(f"({prop_pattern})?")

    # Join patterns with optional commas and any order
    properties_pattern = ",?".join(f"{p}" for p in property_patterns)
    properties_pattern = f"\{{({properties_pattern})\}}"

    # Allow properties to appear in any order
    # properties_pattern = f"^\\{{{properties_pattern}(,?{properties_pattern})*\\}}$"

    return properties_pattern


def validate_json(json_string, regex_pattern):
    try:
        # First, ensure it's valid JSON
        parsed_json = json.loads(json_string)

        # Then, check if it matches our regex pattern
        if regex_pattern.match(json_string):
            return True
        else:
            return False
    except json.JSONDecodeError:
        return False


# Example usage
schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "$id": "https://example.com/employee.schema.json",
    "title": "Record of employee",
    "description": "This document records the details of an employee",
    "type": "object",
    "properties": {
        "id": {"description": "A unique identifier for an employee", "type": "number"},
        "name": {
            "description": "name of the employee",
            "type": "string",
            "minLength": 2,
        },
        "age": {"description": "age of the employee", "type": "number", "minimum": 16},
        "hobbies": {
            "description": "hobbies of the employee",
            "type": "object",
            "properties": {
                "indoor": {
                    "type": "array",
                    "items": {"description": "List of hobbies", "type": "string"},
                    "minItems": 1,
                    "uniqueItems": True,
                },
                "outdoor": {
                    "type": "array",
                    "items": {"description": "List of hobbies", "type": "string"},
                    "minItems": 1,
                    "uniqueItems": True,
                },
            },
            "required": ["indoor", "outdoor"],
        },
    },
    "required": ["id", "name", "age", "hobbies"],
    "additionalProperties": False,
}

regex_pattern = generate_regex_from_schema(schema)
print(regex_pattern)
# # Test cases
# valid_json_1 = (
#     '{"tool": "send_message_to_user", "args": "Hello, world!", "priority": 1}'
# )
# valid_json_2 = (
#     '{"args": "ls -la", "tool": "bash_tool", "timestamp": "2023-05-01T12:00:00Z"}'
# )
# valid_json_3 = '{"tool": "internal_thought", "args": "This is a "quoted" string with \\ backslashes"}'
# invalid_json_1 = '{"tool": "invalid_tool", "args": "Test"}'
# invalid_json_2 = '{"tool": "internal_thought", "invalid_key": "Value"}'

# print(f"Valid JSON 1: {validate_json(valid_json_1, regex_pattern)}")
# print(f"Valid JSON 2: {validate_json(valid_json_2, regex_pattern)}")
# print(f"Valid JSON 3: {validate_json(valid_json_3, regex_pattern)}")
# print(f"Invalid JSON 1: {validate_json(invalid_json_1, regex_pattern)}")
# print(f"Invalid JSON 2: {validate_json(invalid_json_2, regex_pattern)}")
