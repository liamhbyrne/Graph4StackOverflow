from abc import ABC

import networkx as nx
import pandas as pd
from graph4nlp.pytorch.data import GraphData
from graph4nlp.pytorch.modules.graph_construction import DependencyBasedGraphConstruction
from graph4nlp.pytorch.modules.graph_construction.base import StaticGraphConstructionBase
from matplotlib import pyplot as plt


class StaticUserGraphConstruction:
    """Class for StackOverflow user activity graph construction"""

    def __init__(self):
        super(StaticUserGraphConstruction, self).__init__()


    def static_topology(cls, questions: pd.DataFrame, answers: pd.DataFrame, comments: pd.DataFrame) -> GraphData:
        cls._construct_static_graph()

    @classmethod
    def _construct_static_graph(cls, questions: pd.DataFrame, answers: pd.DataFrame, comments: pd.DataFrame):
        user_graph = GraphData()
        next_node = 0

        color_map = []

        node_features = []
        tag_dict = {}  # tag name: node id
        module_dict = {}

        edges_src = []
        edges_dest = []


    @classmethod
    def display_graph(cls, g: GraphData, color_map=None) -> None:
        plt.figure(figsize=(40, 40))
        dgl_ug = g.to_dgl()
        nx_ug_graph = dgl_ug.to_networkx()
        pos_ug = nx.spring_layout(nx_ug_graph)  # , k=0.15, iterations=20)
        if color_map is not None:
            nx.draw(nx_ug_graph, pos_ug, with_labels=True, node_color=color_map)
        else:
            nx.draw(nx_ug_graph, pos_ug, with_labels=True, node_color=[[.7, .7, .7]])


if __name__ == '__main__':
    t = DependencyBasedGraphConstruction(None)
    t(None)

    #graph_topology = StaticUserGraphConstruction()
    #graph_topology(GraphData())
