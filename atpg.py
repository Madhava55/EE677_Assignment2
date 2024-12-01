import numpy as np

class LogicSimulator:
    def __init__(self, circuit_file):
        """
        Initialize the circuit by parsing the input file.
        :param circuit_file: Path to the circuit description text file.
        """
        self.nodes = {}  # Stores node values: {node: value (1, 0, D, D', X)}
        self.gates = []  # List of gates
        self.inputs = []  # List of primary input nodes
        self.outputs = []  # List of primary output nodes

        self.parse_circuit(circuit_file)

        if stuck_val == "sa0":
            self.set_input(stuck_node, "D")  # Faulty node at stuck-at-0
        elif stuck_val == "sa1":
            self.set_input(stuck_node, "D'")  # Faulty node at stuck-at-1
        else:
            raise ValueError("Invalid fault type. Use 'sa0' or 'sa1'.")
        

    def topological_sort(self, gates):
        """Perform a topological sort on the gates to ensure correct simulation order."""
        
        # Create a dictionary to track the in-degree (number of dependencies) of each gate.
        in_degree = {id(gate): 0 for gate in gates}  # Use the gate's id as a unique key
        
        # Create a graph of dependencies.
        for gate in gates:
            for input_node in (gate.get("inputs", [])):
                # Look for the gate that produces the input node.
                for dep_gate in gates:
                    if input_node == dep_gate["output"]:
                        in_degree[id(gate)] += 1
        
        # Queue for gates with no dependencies (in-degree 0).
        queue = [gate for gate in gates if in_degree[id(gate)] == 0]
        
        sorted_gates = []
        
        while queue:
            gate = queue.pop(0)
            sorted_gates.append(gate)
            
            # Decrease the in-degree of dependent gates.
            for dep_gate in gates:
                if gate["output"] in (dep_gate.get("inputs", []) or dep_gate.get("input", [])):
                    in_degree[id(dep_gate)] -= 1
                    if in_degree[id(dep_gate)] == 0:
                        queue.append(dep_gate)
        
        # Return the sorted gates in topological order.
        return sorted_gates
    

    def parse_circuit(self, circuit_file):
        """Parse the circuit from the given text file."""
        with open(circuit_file, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("input"):
                    _, node = line.split()
                    self.inputs.append(node)
                    self.nodes[node] = "X"  # Initialize inputs to "X"
                elif line.startswith("output"):
                    _, node = line.split()
                    self.outputs.append(node)
                    if node not in self.nodes:
                        self.nodes[node] = "X"  # Initialize outputs to "X"
                elif line.startswith("ff"):
                    parts = line.split()
                    gate = {
                        "type": parts[0],
                        "inputs": parts[1],
                        "prev_input": "X",
                        "output": parts[-1]
                    }
                    self.gates.append(gate)
                    if gate["output"] not in self.nodes:
                        self.nodes[gate["output"]] = "X"  # Initialize FFs to "X"
                else:
                    parts = line.split()
                    gate = {
                        "type": parts[0],
                        "inputs": parts[1:-1],
                        "output": parts[-1]
                    }
                    self.gates.append(gate)
                    if gate["output"] not in self.nodes:
                        self.nodes[gate["output"]] = "X"  # Initialize gates to "X"
        self.gates = self.topological_sort(self.gates)

    def set_input(self, node, value):
        """Set the value of an input node."""
        if node in self.nodes:
            self.nodes[node] = value
        else:
            raise ValueError(f"Node {node} not found in the circuit.")

    def simulate_gate(self, gate):
        """Simulate a single gate."""
        inputs = [self.nodes[node] for node in gate["inputs"]]
        if "X" in inputs:  # If any input is unknown
            return "X"

        # Apply gate logic
        if gate["type"] == "and":
            return self.logic_and(inputs)
        elif gate["type"] == "or":
            return self.logic_or(inputs)
        elif gate["type"] == "xor":
            return self.logic_xor(inputs)
        elif gate["type"] == "not":
            return self.logic_not(inputs[0])
        elif gate["type"] == "nand":
            return self.logic_not(self.logic_and(inputs))
        elif gate["type"] == "nor":
            return self.logic_not(self.logic_or(inputs))
        elif gate["type"] == "ff":
            temp = gate["prev_input"]
            gate["prev_input"] = inputs[0]
            return temp
        else:
            raise ValueError(f"Unknown gate type: {gate['type']}")

    def logic_and(self, inputs):
        if "0" in inputs:
            return "0"
        if "D'" in inputs:
            return "D'"
        if "X" in inputs:
            return "X"
        if "D" in inputs:
            return "D"
        return "1"

    def logic_or(self, inputs):
        if "1" in inputs:
            return "1"
        if "D" in inputs:
            return "D"
        if "X" in inputs:
            return "X"
        if "D'" in inputs:
            return "D'"
        return "0"

    def logic_xor(self, inputs):
        if inputs[0] == inputs[1]:
            return "0"
        if inputs[0] == "1" and inputs[1] == "D'":
            return "1"
        if inputs[0] == "0" and inputs[1] == "D":
            return "1"
        if "X" in inputs:
            return "X"
        return "0"

    def logic_not(self, input):
        if input == "1":
            return "0"
        if input == "0":
            return "1"
        if input == "D":
            return "D'"
        if input == "D'":
            return "D"
        return "X"

    def run(self):
        """Run the simulation."""
        for gate in self.gates:
            if gate["output"] == stuck_node:
                continue  # Skip updating the stuck node
            self.nodes[gate["output"]] = self.simulate_gate(gate)

    def get_node_value(self, node):
        """Get the value of a node."""
        if node in self.nodes:
            return self.nodes[node]
        else:
            raise ValueError(f"Node {node} not found in the circuit.")
    
    def calculate_node_value(self, node):
        """Calculate the value of a node based on its inputs (without modifying the node)."""
        if node not in self.nodes:
            raise ValueError(f"Node {node} not found in the circuit.")
        
        # Get the corresponding gate for the node
        for gate in self.gates:
            if gate["output"] == node:
                # Simulate the gate logic but do not update the node value
                inputs = [self.nodes[input_node] for input_node in gate["inputs"]]
                if gate["type"] == "and":
                    return self.logic_and(inputs)
                elif gate["type"] == "or":
                    return self.logic_or(inputs)
                elif gate["type"] == "xor":
                    return self.logic_xor(inputs)
                elif gate["type"] == "not":
                    return self.logic_not(inputs[0])
                elif gate["type"] == "nand":
                    return self.logic_not(self.logic_and(inputs))
                elif gate["type"] == "nor":
                    return self.logic_not(self.logic_or(inputs))
                elif gate["type"] == "ff":
                    return self.logic_ff(inputs[0], gate["prev_input"])
                else:
                    raise ValueError(f"Unknown gate type: {gate['type']}")
        
        # If no gate is found for the node, return "X" as default
        return "X"

import numpy as np

def generate_binary_matrices(m, n):
    # Total number of binary matrices is 2^(m * n)
    total_matrices = 2 ** (m * n)
    
    matrices = []
    
    for i in range(total_matrices):
        # Convert i to a binary string of length m*n
        bin_str = bin(i)[2:].zfill(m * n)  # Remove '0b' prefix and pad to the required length
        # Convert the binary string to a 2D matrix with "1" and "0" as string elements
        matrix = np.array(list(bin_str), dtype=str).reshape(m, n)
        # Replace '0' and '1' with string "0" and "1"
        matrix = np.where(matrix == '0', "0", "1")
        matrices.append(matrix)
    
    return matrices


def check_fault(simulator):
    done = 1
    done = 1
    stuck_node_val = simulator.calculate_node_value(stuck_node)
    output_value = simulator.get_node_value(output_node)
    if stuck_val == "sa0" and stuck_node_val != "1":
        done = 0
    if stuck_val == "sa1" and stuck_node_val != "0":
        done = 0
    if output_value not in ["D", "D'"]:
        done = 0
    return done


def test_fault():
    """Test the fault by trying all input configurations."""
    simulator = LogicSimulator("circuit.txt")
    print(ff_count)
    testvec = generate_binary_matrices(ff_count+1,len(simulator.inputs))
    for matrix in testvec:
        simulator = LogicSimulator("circuit.txt")
        # Generate the binary input configuration
        for i in range(ff_count+1):
            # Set input values
            input_values = matrix[i].tolist()
            for idx, node in enumerate(simulator.inputs):
                simulator.set_input(node, input_values[idx])
            simulator.run()
            # Check if the fault is detectable
        if check_fault(simulator):
            print(f"Fault detected! Configuration {matrix} makes the fault testable.")
            return True
    
    print("Fault is not testable.")
    return False  
stuck_node = input("Please enter the stuck-at-fault node")
stuck_val = input("Please enter the type of fault , sa0 or sa1")
output_node = input("Enter the output node")
ff_count = int(input("Enter sequential depth"))
test_fault()
