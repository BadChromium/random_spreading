import numpy as np
import igraph as ig

class factor:
    def __init__(self, variables = None, distribution = None):
        if (distribution is None) and (variables is not None):
            self.__set_data(np.array(variables), None, None)
        elif (variables is None) or (len(variables) != len(distribution.shape)):
            raise Exception('Data is incorrect')
        else:
            self.__set_data(np.array(variables),
                            np.array(distribution),
                            np.array(distribution.shape))
    
    def __set_data(self, variables, distribution, shape):
        self.__variables    = variables
        self.__distribution = distribution
        self.__shape        = shape
    
    # ----------------------- Info --------------------------
    def is_none(self):
        return True if self.__distribution is None else False
        
    # ----------------------- Getters -----------------------
    def get_variables(self):
        return self.__variables
    
    def get_distribution(self):
        return self.__distribution
    
    def get_shape(self):
        return self.__shape

class factor_graph:
    def __init__(self):
        self._graph = ig.Graph()

    def get_node_status(self, name):
        if len(self._graph.vs) == 0:
            return False
        elif len(self._graph.vs.select(name_eq=name)) == 0:
            return False
        else:
            if self._graph.vs.find(name=name)['is_factor'] == True:
                return 'factor'
            else:
                return 'variable'

    def __check_variable_ranks(self, f_name, factor_, allowded_v_degree):
        for counter, v_name in enumerate(factor_.get_variables()):
            if (self.get_node_status(v_name) == 'variable') and (not factor_.is_none()):
                if     (self._graph.vs.find(name=v_name)['rank'] != factor_.get_shape()[counter]) \
                and (self._graph.vs.find(name=v_name)['rank'] != None) \
                and (self._graph.vs.find(v_name).degree() > allowded_v_degree):
                    raise Exception('Invalid shape of factor_')

    def __set_variable_ranks(self, f_name, factor_):
        for counter, v_name in enumerate(factor_.get_variables()):
            if factor_.is_none():
                self._graph.vs.find(name=v_name)['rank'] = None
            else:
                self._graph.vs.find(name=v_name)['rank'] = factor_.get_shape()[counter]       

    # -------------factor node functions -------------
    def add_factor_node(self, f_name, factor_):
        if (self.get_node_status(f_name) != False) or (f_name in factor_.get_variables()):
            raise Exception('Invalid factor name')
        if type(factor_) is not factor:
            raise Exception('Invalid factor_')
        for v_name in factor_.get_variables():
            if self.get_node_status(v_name) == 'factor':
                raise Exception('Invalid factor')
    
        # Check ranks
        self.__check_variable_ranks(f_name, factor_, 1)
        # Create variables
        for v_name in factor_.get_variables():
            if self.get_node_status(v_name) == False:
                self.__create_variable_node(v_name)
        # Set ranks
        self.__set_variable_ranks(f_name, factor_)
        # Add node and corresponding edges
        self.__create_factor_node(f_name, factor_)

    def change_factor_distribution(self, f_name, factor_):
        if self.get_node_status(f_name) != 'factor':
            raise Exception('Invalid variable name')
        if set(factor_.get_variables()) != set(self._graph.vs[self._graph.neighbors(f_name)]['name']):
            raise Exception('invalid factor distribution')
        
        # Check ranks
        self.__check_variable_ranks(f_name, factor_, 0)
        # Set ranks
        self.__set_variable_ranks(f_name, factor_)
        # Set data
        self._graph.vs.find(name=f_name)['factor_'] = factor_

    def remove_factor(self, f_name, remove_zero_degree=False):
        if self.get_node_status(f_name) != 'factor':
            raise Exception('Invalid variable name')
        
        neighbors = self._graph.neighbors(f_name, mode="out")
        self._graph.delete_vertices(f_name)
        
        if remove_zero_degree:
            for v_name in neighbors:
                if self._graph.vs.find(v_name).degree() == 0:
                    self.remove_variable(v_name)

    def __create_factor_node(self, f_name, factor_):
        # Create node
        self._graph.add_vertex(f_name)
        self._graph.vs.find(name=f_name)['is_factor'] = True
        self._graph.vs.find(name=f_name)['factor_']   = factor_
        
        # Create corresponding edges
        start = self._graph.vs.find(name=f_name).index
        edge_list = [tuple([start, self._graph.vs.find(name=i).index]) for i in factor_.get_variables()]
        self._graph.add_edges(edge_list)

    # ------------variable node functions-------------
    def add_variable_node(self, v_name):
        if self.get_node_status(v_name) != False:
            raise Exception('Node already exists')
        self.__create_variable_node(v_name)

    def remove_variable(self, v_name):
        if self.get_node_status(v_name) != 'variable':
            raise Exception('Invalid variable name')
        if self._graph.vs.find(v_name).degree() != 0:
            raise Exception('Can not delete variables with degree >0')
        self._graph.delete_vertices(self._graph.vs.find(v_name).index)
    
    def __create_variable_node(self, v_name, rank=None):
        self._graph.add_vertex(v_name)
        self._graph.vs.find(name=v_name)['is_factor'] = False
        self._graph.vs.find(name=v_name)['rank'] = rank

    def get_graph(self):
        return self._graph

    def is_connected(self):
        return self._graph.is_connected()

    def is_loop(self):
        return any(self._graph.is_loop())