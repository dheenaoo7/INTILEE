import json
import os
from tree_sitter import Language, Parser

# Build the Tree-Sitter Java language library
Language.build_library(
    'my-languages.so',  # Output file
    ['/opt/tree-sitter-java']  # Path to the tree-sitter-java repository
)

# Load the parser with Java grammar
JAVA_LANGUAGE = Language('/home/dheena/Downloads/Intiliee/my-languages.so', 'java')
parser = Parser()
parser.set_language(JAVA_LANGUAGE)

def parse_file(file_path):
    """Parse a Java file and return the root node of the AST."""
    try:
        with open(file_path, 'r') as file:
            code = file.read()

        # Parsing the code using the Tree-Sitter parser
        tree = parser.parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        return root_node, code
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None, None

def extract_functions_and_classes(node, code):
    """Recursively extract function and class names from the AST."""
    functions = []
    classes = []
    functions_code = []
    classes_code = []

    try:
        for child in node.children:
            if child.type == 'method_declaration':
                # Extracting function name
                function_name = code[child.child_by_field_name('name').start_byte:child.child_by_field_name('name').end_byte]
                functions.append(function_name)
                
                # Extracting function code snippet and add to list
                function_code = code[child.start_byte:child.end_byte]
                functions_code.append(function_code)

            elif child.type == 'class_declaration':
                # Extracting class name
                class_name = code[child.child_by_field_name('name').start_byte:child.child_by_field_name('name').end_byte]
                classes.append(class_name)
                
                # Extracting class code snippet and add to list
                class_code = code[child.start_byte:child.end_byte]
                classes_code.append(class_code)

            # Recursively process child nodes
            sub_functions, sub_classes, sub_functions_code, sub_classes_code = extract_functions_and_classes(child, code)
            functions.extend(sub_functions)
            classes.extend(sub_classes)
            functions_code.extend(sub_functions_code)
            classes_code.extend(sub_classes_code)

    except Exception as e:
        print(f"Error extracting functions and classes: {e}")

    return functions, classes, functions_code, classes_code

def parse_codebase(directory):
    """Parse an entire Java codebase to extract functions and classes."""
    all_files_data = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                print(f"Parsing {file_path}...")

                # Parse the file and extract code
                root_node, code = parse_file(file_path)
                if root_node:
                    functions, classes, functions_code, classes_code = extract_functions_and_classes(root_node, code)

                    # Structure the data
                    file_data = {
                        'file_name': file_path,
                        'functions': functions,
                        'classes': classes,
                        'code_snippets': {
                            'functions': functions_code,  # Comma-separated individual function snippets
                            'classes': classes_code
                        }
                    }
                    all_files_data.append(file_data)

    return all_files_data

def save_to_json(data, output_path):
    """Save the extracted data to a JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")

# Example: Parse the 'my_java_project' directory
directory_path = '/home/dheena/workspace/microservices/ticket-configuration-microservice'
all_code_data = parse_codebase(directory_path)

# Save the extracted data to a JSON file
save_to_json(all_code_data, '/home/dheena/Downloads/Intiliee/output/output_code_data.json')
