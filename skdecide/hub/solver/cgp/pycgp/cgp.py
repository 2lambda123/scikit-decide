# Copyright (c) AIRBUS and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random as rnd
import sys

import numpy as np


class CGP:
    class CGPFunc:
        def __init__(self, f, name, arity):
            self.function = f
            self.name = name
            self.arity = arity

    class CGPNode:
        def __init__(self, args, f):
            self.args = args
            self.function = f

    def __init__(
        self,
        genome,
        num_inputs,
        num_outputs,
        num_cols,
        num_rows,
        library,
        recurrency_distance=1.0,
    ):
        """Creates a new instance of the Graph class.
        Parameters:
            - genome (dict): A dictionary representing the genome of the graph.
            - num_inputs (int): The number of input nodes in the graph.
            - num_outputs (int): The number of output nodes in the graph.
            - num_cols (int): The number of columns in the graph.
            - num_rows (int): The number of rows in the graph.
            - library (list): A list of functions that can be used in the graph.
            - recurrency_distance (float): The maximum distance between recurrent connections.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - Copies the genome to the instance variable.
            - Sets the number of inputs, outputs, columns, and rows.
            - Calculates the maximum graph length.
            - Sets the library and maximum arity.
            - Sets the graph_created flag to False."""

        self.genome = genome.copy()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.max_graph_length = num_cols * num_rows
        self.library = library
        self.max_arity = 0
        self.recurrency_distance = recurrency_distance
        for f in self.library:
            self.max_arity = np.maximum(self.max_arity, f.arity)
        self.graph_created = False

    def create_graph(self):
        """Creates a computational graph for CGP.
        Parameters:
            - self (CGP): The CGP object.
        Returns:
            - None: The function modifies the CGP object in-place.
        Processing Logic:
            - Initializes necessary arrays.
            - Extracts output genes from genome.
            - Builds node list from genome.
            - Calls node_to_evaluate().
            - Sets graph_created flag to True."""

        self.to_evaluate = np.zeros(self.max_graph_length, dtype=bool)
        self.node_output = np.zeros(
            self.max_graph_length + self.num_inputs, dtype=np.float64
        )
        self.nodes_used = []
        self.output_genes = np.zeros(self.num_outputs, dtype=np.int)
        self.nodes = np.empty(0, dtype=self.CGPNode)
        for i in range(0, self.num_outputs):
            self.output_genes[i] = self.genome[len(self.genome) - self.num_outputs + i]
        i = 0
        # building node list
        while i < len(self.genome) - self.num_outputs:
            f = self.genome[i]
            args = np.empty(0, dtype=int)
            for j in range(self.max_arity):
                args = np.append(args, self.genome[i + j + 1])
            i += self.max_arity + 1
            self.nodes = np.append(self.nodes, self.CGPNode(args, f))
        self.node_to_evaluate()
        self.graph_created = True

    def node_to_evaluate(self):
        """This function determines which nodes need to be evaluated in a genetic algorithm.
        Parameters:
            - self (object): The object containing the genetic algorithm.
        Returns:
            - np.array: An array of the nodes that need to be evaluated.
        Processing Logic:
            - Determines which nodes need to be evaluated.
            - Sets the corresponding index in the to_evaluate array to True.
            - Adds the current node to the nodes_used array.
            - Decrements the index by 1.
            - Increments the index by 1."""

        p = 0
        while p < self.num_outputs:
            if self.output_genes[p] - self.num_inputs >= 0:
                self.to_evaluate[self.output_genes[p] - self.num_inputs] = True
            p = p + 1
        p = self.max_graph_length - 1
        while p >= 0:
            if self.to_evaluate[p]:
                for i in range(0, len(self.nodes[p].args)):
                    arg = self.nodes[p].args[i]
                    if arg - self.num_inputs >= 0:
                        self.to_evaluate[arg - self.num_inputs] = True
                self.nodes_used.append(p)
            p = p - 1
        self.nodes_used = np.array(self.nodes_used)

    def load_input_data(self, input_data):
        """Loads input data into the node_output list.
        Parameters:
            - input_data (list): List of input data.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - Loop through num_inputs.
            - Set node_output[p] to input_data[p]."""

        for p in range(self.num_inputs):
            self.node_output[p] = input_data[p]

    def compute_graph(self):
        """Computes the output of the graph by evaluating each node's function using its arguments and updating the node's output value.
        Parameters:
            - self (object): The current instance of the class.
        Returns:
            - None: The function does not return any value, but updates the node_output attribute of the class.
        Processing Logic:
            - Updates the node_output attribute by evaluating each node's function.
            - Stores the previous node_output in the node_output_old attribute.
            - Uses the library attribute to access the function of each node.
            - Checks for any NaN or out of range values in the node_output.
        Example:
            compute_graph(self)
            # Updates the node_output attribute of the class."""

        self.node_output_old = self.node_output.copy()
        p = len(self.nodes_used) - 1
        while p >= 0:
            args = np.zeros(self.max_arity)
            for i in range(0, self.max_arity):
                args[i] = self.node_output_old[self.nodes[self.nodes_used[p]].args[i]]
            f = self.library[self.nodes[self.nodes_used[p]].function].function
            self.node_output[self.nodes_used[p] + self.num_inputs] = f(args)

            if (
                self.node_output[self.nodes_used[p] + self.num_inputs]
                != self.node_output[self.nodes_used[p] + self.num_inputs]
            ):
                print(
                    self.library[self.nodes[self.nodes_used[p]].function].name,
                    " returned NaN with ",
                    args,
                )
            if (
                self.node_output[self.nodes_used[p] + self.num_inputs] < -1.0
                or self.node_output[self.nodes_used[p] + self.num_inputs] > 1.0
            ):
                print(
                    self.library[self.nodes[self.nodes_used[p]].function].name,
                    " returned ",
                    self.node_output[self.nodes_used[p] + self.num_inputs],
                    " with ",
                    args,
                )

            p = p - 1

    def run(self, inputData):
        """"Runs the graph computation on the provided input data and returns the computed output.
        Parameters:
            - inputData (dict): A dictionary containing the input data for the graph computation.
        Returns:
            - dict: A dictionary containing the computed output from the graph computation.
        Processing Logic:
            - Creates graph if not already created.
            - Loads input data into the graph.
            - Computes the graph.
            - Reads and returns the output from the graph computation.""""

        if not self.graph_created:
            self.create_graph()

        self.load_input_data(inputData)
        self.compute_graph()
        return self.read_output()

    def read_output(self):
        """Reads the output values from the neural network.
        Parameters:
            - self (object): The neural network object.
        Returns:
            - output (numpy array): An array containing the output values.
        Processing Logic:
            - Create an array of zeros.
            - Loop through the output genes.
            - Assign the output values to the array.
            - Return the array."""

        output = np.zeros(self.num_outputs)
        for p in range(0, self.num_outputs):
            output[p] = self.node_output[self.output_genes[p]]
        return output

    def clone(self):
        """Creates a clone of the current CGP object.
        Parameters:
            - self (CGP): The current CGP object.
        Returns:
            - CGP: A clone of the current CGP object.
        Processing Logic:
            - Create a new CGP object.
            - Copy the genome, number of inputs, number of outputs, number of columns, number of rows, and library from the current CGP object.
            - Return the new CGP object.
        Example:
            cgp = CGP()
            cgp_clone = cgp.clone()
            # cgp_clone is now a clone of cgp"""

        return CGP(
            self.genome,
            self.num_inputs,
            self.num_outputs,
            self.num_cols,
            self.num_rows,
            self.library,
        )

    def mutate(self, num_mutationss):
        """Mutates the genome by randomly changing the values of the nodes and connections.
        Parameters:
            - self (Genome): The genome to be mutated.
            - num_mutations (int): The number of mutations to be performed.
        Returns:
            - None: The function does not return anything, it simply mutates the genome in place.
        Processing Logic:
            - Randomly select an index in the genome.
            - If the index corresponds to an internal node, mutate the function or connection.
            - If the index corresponds to an output node, mutate the connection.
            - Repeat for the specified number of mutations."""

        for i in range(0, num_mutationss):
            index = rnd.randint(0, len(self.genome) - 1)
            if index < self.num_cols * self.num_rows * (self.max_arity + 1):
                # this is an internal node
                if index % (self.max_arity + 1) == 0:
                    # mutate function
                    self.genome[index] = rnd.randint(0, len(self.library) - 1)
                else:
                    # mutate connection
                    self.genome[index] = rnd.randint(
                        0,
                        self.num_inputs
                        + (int(index / (self.max_arity + 1)) - 1) * self.num_rows,
                    )
            else:
                # this is an output node
                self.genome[index] = rnd.randint(
                    0, self.num_inputs + self.num_cols * self.num_rows - 1
                )

    def mutate_per_gene(self, mutation_rate_nodes, mutation_rate_outputs):
        """Mutates the genome of an individual by randomly changing the function or connection of each internal node and output node based on the given mutation rates.
        Parameters:
            - mutation_rate_nodes (float): The probability of mutating a function or connection in an internal node.
            - mutation_rate_outputs (float): The probability of mutating a connection in an output node.
        Returns:
            - None: The genome of the individual is mutated in-place.
        Processing Logic:
            - Mutates internal nodes and output nodes separately.
            - Mutates functions and connections separately.
            - Only mutates if the random number generated is less than the given mutation rate."""

        for index in range(0, len(self.genome)):
            if index < self.num_cols * self.num_rows * (self.max_arity + 1):
                # this is an internal node
                if rnd.random() < mutation_rate_nodes:
                    if index % (self.max_arity + 1) == 0:
                        # mutate function
                        self.genome[index] = rnd.randint(0, len(self.library) - 1)
                    else:
                        # mutate connection
                        self.genome[index] = rnd.randint(
                            0,
                            min(
                                self.max_graph_length + self.num_inputs - 1,
                                (
                                    self.num_inputs
                                    + (int(index / (self.max_arity + 1)) - 1)
                                    * self.num_rows
                                )
                                * self.recurrency_distance,
                            ),
                        )
                        # self.genome[index] = rnd.randint(0, self.num_inputs + (int(index / (self.max_arity + 1)) - 1) * self.num_rows)
            else:
                # this is an output node
                if rnd.random() < mutation_rate_outputs:
                    # this is an output node
                    self.genome[index] = rnd.randint(
                        0, self.num_inputs + self.num_cols * self.num_rows - 1
                    )

    def to_dot(self, file_name, input_names, output_names):
        """to_dot(self, file_name, input_names, output_names):
            Creates a .dot file of the graph of the Cartesian Genetic Programming (CGP) algorithm.
            Parameters:
                - file_name (str): The name of the .dot file to be created.
                - input_names (list): A list of strings containing the names of the input nodes.
                - output_names (list): A list of strings containing the names of the output nodes.
            Returns:
                - None: The function does not return any value.
            Processing Logic:
                - Creates a graph if one has not already been created.
                - Opens the specified file in write mode.
                - Writes the initial lines of the .dot file.
                - Initializes an empty array to keep track of visited nodes.
                - Writes the output nodes to the .dot file.
                - Calls the _write_dot_from_gene function for each output node.
                - Closes the file."""

        if not self.graph_created:
            self.create_graph()
        out = open(file_name, "w")
        out.write("digraph cgp {\n")
        out.write('\tsize = "4,4";\n')
        self.dot_rec_visited_nodes = np.empty(1)
        for i in range(self.num_outputs):
            out.write("\t" + output_names[i] + " [shape=oval];\n")
            self._write_dot_from_gene(
                output_names[i], self.output_genes[i], out, 0, input_names, output_names
            )
        out.write("}")
        out.close()

    def _write_dot_from_gene(self, to_name, pos, out, a, input_names, output_names):
        """Function that writes a dot file from a gene.
        Parameters:
            - to_name (str): Name of the gene.
            - pos (int): Position of the gene.
            - out (file): Output file.
            - a (int): Argument.
            - input_names (list): List of input names.
            - output_names (list): List of output names.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - Write input names and connections.
            - Write output names and connections.
            - Keep track of visited nodes.
            - Recursively call function for each argument."""

        if pos < self.num_inputs:
            out.write("\t" + input_names[pos] + " [shape=polygon,sides=6];\n")
            out.write(
                "\t"
                + input_names[pos]
                + " -> "
                + to_name
                + ' [label="'
                + str(a)
                + '"];\n'
            )
            self.dot_rec_visited_nodes = np.append(self.dot_rec_visited_nodes, [pos])
        else:
            pos -= self.num_inputs
            out.write(
                "\t"
                + self.library[self.nodes[pos].function].name
                + "_"
                + str(pos)
                + " -> "
                + to_name
                + ' [label="'
                + str(a)
                + '"];\n'
            )
            if pos + self.num_inputs not in self.dot_rec_visited_nodes:
                out.write(
                    "\t"
                    + self.library[self.nodes[pos].function].name
                    + "_"
                    + str(pos)
                    + " [shape=none];\n"
                )
                for a in range(self.library[self.nodes[pos].function].arity):
                    self._write_dot_from_gene(
                        self.library[self.nodes[pos].function].name + "_" + str(pos),
                        self.nodes[pos].args[a],
                        out,
                        a,
                        input_names,
                        output_names,
                    )
            self.dot_rec_visited_nodes = np.append(
                self.dot_rec_visited_nodes, [pos + self.num_inputs]
            )

    def to_function_string(self, input_names, output_names):
        """Converts a given set of input and output names into a string representation of a function.
        Parameters:
            - input_names (list): List of input names.
            - output_names (list): List of output names.
        Returns:
            - function_string (str): String representation of the function.
        Processing Logic:
            - Creates a graph if one does not already exist.
            - Prints each output name followed by an equals sign.
            - Writes the function expression for each output gene using the input and output names.
            - Prints a semicolon and a new line after each output gene."""

        if not self.graph_created:
            self.create_graph()
        for o in range(self.num_outputs):
            print(output_names[o] + " = ", end="")
            self._write_from_gene(self.output_genes[o], input_names, output_names)
            print(";")
            print("")

    def _write_from_gene(self, pos, input_names, output_names):
        """"Prints the expression represented by the gene at the given position, using the provided input and output names."
        Parameters:
            - pos (int): The position of the gene to be evaluated.
            - input_names (list): A list of input names.
            - output_names (list): A list of output names.
        Returns:
            - None: This function does not return any value.
        Processing Logic:
            - Prints the input name at the given position if it is less than the number of inputs.
            - Otherwise, prints the function name followed by its arguments in parentheses.
            - Recursively calls itself to print the arguments.
            - Each argument is separated by a comma, except for the last one.
            - The entire expression is printed without any spaces."""

        if pos < self.num_inputs:
            print(input_names[pos], end="")
        else:
            pos -= self.num_inputs
            print(self.library[self.nodes[pos].function].name + "(", end="")
            for a in range(self.library[self.nodes[pos].function].arity):
                # print(' ', end='')
                self._write_from_gene(
                    self.nodes[pos].args[a], input_names, output_names
                )
                if a != self.library[self.nodes[pos].function].arity - 1:
                    print(", ", end="")
                # else:
                # 	print(')', end='')
            print(")", end="")

    @classmethod
    def random(
        cls, num_inputs, num_outputs, num_cols, num_rows, library, recurrency_distance
    ):
        """This function generates a random Compositional Pattern-Producing Network (CGP) genome based on the given parameters.
        Parameters:
            - cls (class): The class to be used for the CGP object.
            - num_inputs (int): The number of input nodes in the CGP.
            - num_outputs (int): The number of output nodes in the CGP.
            - num_cols (int): The number of columns in the CGP.
            - num_rows (int): The number of rows in the CGP.
            - library (list): A list of functions to be used in the CGP.
            - recurrency_distance (int): The maximum distance between recurrent connections in the CGP.
        Returns:
            - CGP: A randomly generated CGP object.
        Processing Logic:
            - Finds the maximum arity (number of inputs) among the functions in the library.
            - Creates a genome array with enough space for all nodes and connections.
            - Iterates through each column and row in the CGP, randomly selecting a function and its inputs.
            - Adds the selected function and inputs to the genome array.
            - Iterates through each output node, randomly selecting its input.
            - Returns a CGP object with the generated genome and given parameters."""

        max_arity = 0
        for f in library:
            max_arity = np.maximum(max_arity, f.arity)
        genome = np.zeros(
            num_cols * num_rows * (max_arity + 1) + num_outputs, dtype=int
        )
        gPos = 0
        for c in range(0, num_cols):
            for r in range(0, num_rows):
                genome[gPos] = rnd.randint(0, len(library) - 1)
                for a in range(max_arity):
                    genome[gPos + a + 1] = rnd.randint(0, num_inputs + c * num_rows - 1)
                gPos = gPos + max_arity + 1
        for o in range(0, num_outputs):
            genome[gPos] = rnd.randint(0, num_inputs + num_cols * num_rows - 1)
            gPos = gPos + 1
        return CGP(
            genome,
            num_inputs,
            num_outputs,
            num_cols,
            num_rows,
            library,
            recurrency_distance,
        )

    def save(self, file_name):
        """Saves the current state of the network to a file.
        Parameters:
            - file_name (str): The name of the file to save the network to.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - Open the specified file for writing.
            - Write the number of inputs, outputs, columns, and rows to the file.
            - Write each genome in the network to the file.
            - Write each function in the library to the file."""

        out = open(file_name, "w")
        out.write(str(self.num_inputs) + " ")
        out.write(str(self.num_outputs) + " ")
        out.write(str(self.num_cols) + " ")
        out.write(str(self.num_rows) + "\n")
        for g in self.genome:
            out.write(str(g) + " ")
        out.write("\n")
        for f in self.library:
            out.write(f.name + " ")
        out.close()

    @classmethod
    def load_from_file(cls, file_name, library):
        """This function loads data from a file and creates a CGP object using the data.
        Parameters:
            - cls (class): The class of the CGP object.
            - file_name (str): The name of the file to load data from.
            - library (str): The library to use for the CGP object.
        Returns:
            - CGP: A CGP object created using the data from the file.
        Processing Logic:
            - Open the file and read the first 5 million characters.
            - Split the data into three lists: pams, genes, and funcs.
            - Close the file.
            - Create an empty numpy array to store the parameters.
            - Loop through the pams list and convert each element to an integer before appending it to the params array.
            - Create an empty numpy array to store the genome.
            - Loop through the genes list and convert each element to an integer before appending it to the genome array.
            - Create a CGP object using the genome, params, and library."""

        inp = open(file_name, "r")
        pams = inp.readline(5_000_000).split()
        genes = inp.readline(5_000_000).split()
        funcs = inp.readline(5_000_000).split()
        inp.close()
        params = np.empty(0, dtype=int)
        for p in pams:
            params = np.append(params, int(p))
        genome = np.empty(0, dtype=int)
        for g in genes:
            genome = np.append(genome, int(g))
        return CGP(genome, params[0], params[1], params[2], params[3], library)

    @classmethod
    def test(cls, num):
        """Calculates the mutation of a given number of genomes and prints the mutated genome and the result of running the mutated genome on a given input.
        Parameters:
            - cls (CGP): An instance of the CGP class.
            - num (int): The number of genomes to mutate.
        Returns:
            - None: This function does not return any value.
        Processing Logic:
            - Generate a random genome using the CGP class.
            - For each genome in the range of 0 to num, mutate the genome.
            - Print the mutated genome.
            - Print the result of running the mutated genome on a given input.
        Example:
            test(CGP, 5)
            # Output:
            # [1, 1, 1, 1, 1, 1, 1, 1, 1]
            # 3"""

        c = CGP.random(2, 1, 2, 2, 2)
        for i in range(0, num):
            c.mutate(1)
            print(c.genome)
            print(c.run([1, 2]))
