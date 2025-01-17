import javalang
import os
import json

class Methods:
    classname = ""
    methodname = ""
    parameters = []
    def __init__(self, classname, methodname, parameters):
        self.classname = classname
        self.methodname = methodname
        self.parameters = parameters    
    def is_equal(self, method):
        return (
            self.classname == method.classname and
            self.methodname == method.methodname and
            self.parameters == method.parameters
        ) 
    def to_json(self):
        return {
            "classname": self.classname,
            "methodname": self.methodname,
            "parameters": list(self.parameters)
        }   
    def __hash__(self):
        return hash((self.classname, self.methodname, tuple(self.parameters)))    
    def __eq__(self, other):
        return self.is_equal(other)    
        
unique_methods = []
def save_methods_to_json(methods, output_file):
    """Convert method data to JSON and save to a file."""
    json_data = [{"classname": method.classname, "methodname": method.methodname, "parameters": method.parameters} for method in methods]
    
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Data saved to {output_file}")

def extract_unique_methods_from_ast(file_path):
    with open(file_path, 'r') as file:
        code = file.read()
    tree = javalang.parse.parse(code)
    # Traverse the AST to find method declarations
    current_class_name = ""
    for _, node in tree:    
        if isinstance(node, javalang.tree.ClassDeclaration):
                # Extract the class name when encountering a class declaration
            current_class_name = node.name
        if isinstance(node, javalang.tree.MethodDeclaration):
            # Extract the method name 
            method_name = node.name
            # Extract the parameter type
            param_types = []
            for param in node.parameters:
                param_type = param.type
                param_types.append(str(param_type))  
            unique_methods.append(Methods(current_class_name, method_name, tuple(param_types)))
    save_methods_to_json(unique_methods, '/home/dheena/Downloads/Intiliee/output/unique_methods.json')
def parse_java_file(file_path):
    """Parse a single Java file and extract function details."""
    with open(file_path, 'r') as file:
        code = file.read()

    code_lines = code.splitlines()
    tree = javalang.parse.parse(code)

    functions_data = []
    for _, node in tree.filter(javalang.tree.MethodDeclaration):
        function_data = {
            "name": node.name,
            "param": [f"{p.type.name} {p.name}" for p in node.parameters],
            "calls": set()
        }
        if node.position:
            start_line = node.position[0] - 1  # Convert 1-based to 0-based indexing
            end_line = None
            if node.body:
                end_line = node.body[-1].position[0]  # Last statement's line
            else:
                end_line = start_line + 1  # Assume a single line if the body is empty

            method_code = "\n".join(code_lines[start_line:end_line + 1])
            function_data["code"] = method_code.strip()  # Clean whitespace
        else:
            function_data["code"] = ""

        # Traverse method body to find method invocations
        if node.body:
            for child in node.body:
                if isinstance(child, javalang.ast.Node):
                    for _, call_node in child.filter(javalang.tree.MethodInvocation):
                        call_class = call_node.qualifier if call_node.qualifier else "this"
                        call_name = call_node.member
                        call_args = tuple(str(arg) for arg in call_node.arguments)
                          # Create a Methods object for the call
                        method_call = Methods(call_class, call_name, call_args)
                        function_data["calls"].add(method_call)


        functions_data.append(function_data)

    return functions_data


def parse_codebase(directory):
    """Parse an entire Java codebase to extract functions and classes."""
    all_files_data = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                extract_unique_methods_from_ast(os.path.join(root, file))
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                print(f"Parsing {file_path}...")
                file_data = parse_java_file(file_path)
                all_files_data.extend(file_data)

    return all_files_data

def save_to_json(data, output_path):
    """Save the extracted data to a JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {output_path}")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")

directory_path = '/home/dheena/workspace/microservices/ticket-configuration-microservice'
all_code_data = parse_codebase(directory_path)
for data in all_code_data:
    data["calls"] = [method.to_json() for method in data["calls"]]

# Save the extracted data to a JSON file
save_to_json(all_code_data, '/home/dheena/Downloads/Intiliee/output/output_code_data.json')