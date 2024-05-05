import networkx as nx
from skimage.transform import EuclideanTransform
from data_structures.contour import Contour
from data_structures.contour_matching_results import ContourMatchingResults
from networkx.utils.union_find import UnionFind


def merge_pose_forest_nodes(
    pose_forest: nx.DiGraph,
    source_node: int,
    target_node: int,
    relative_pose: EuclideanTransform,
) -> None:
    """
    Merges two nodes in a pose forest by adjusting the relative poses in the graph.
    The function re-parents the target node and its ancestors, if any, to the source
    node using the given relative pose. Modifications are made in-place on the pose
    forest.

    :param pose_forest:
        The directed graph representing the pose forest. It must follow the structure
        of a directed acyclic graph (DAG) where nodes represent poses, and edges are
        annotated with `relative_pose` transformations.
    :param source_node:
        The node in the pose forest to which the other node will be merged.
    :param target_node:
        The node in the pose forest that will be re-parented to the source node.
    :param relative_pose:
        The relative transformation from `source_node` to `target_node`.
    :return:
        This function modifies `pose_forest` in place and does not return any value.

    """
    assert 0 <= source_node < pose_forest.number_of_nodes()
    assert 0 <= target_node < pose_forest.number_of_nodes()
    assert source_node != target_node

    curr_node = target_node
    # Save the edges to be re-added later
    edges_buffer: list[tuple[int, int, EuclideanTransform]] = []
    while True:
        parent_nodes = list(pose_forest.predecessors(curr_node))
        if not parent_nodes:
            break
        assert len(parent_nodes) == 1
        parent_node = parent_nodes[0]
        parent_relative_pose = pose_forest.get_edge_data(parent_node, curr_node)[
            "relative_pose"
        ].inverse
        pose_forest.remove_edge(parent_node, curr_node)
        edges_buffer.append((curr_node, parent_node, parent_relative_pose))
        curr_node = parent_node
    for curr_node, parent_node, parent_relative_pose in edges_buffer:
        pose_forest.add_edge(curr_node, parent_node, relative_pose=parent_relative_pose)
    pose_forest.add_edge(source_node, target_node, relative_pose=relative_pose)


class IncrementalMatching:
    def __init__(
        self,
        contour_graph: nx.Graph,
        num_puzzle_pieces: int,
        contours: list[Contour],
    ) -> None:
        assert num_puzzle_pieces > 0
        assert len(contours) > 0

        self._contour_graph_edges = sorted(
            contour_graph.edges(data="matching_info"),
            key=lambda edge: -len(
                edge[2].refined_matching_result.source_matching_indices
            )
            - len(edge[2].refined_matching_result.target_matching_indices),
        )  # All edges (n1, n2, matching_info) sorted by the matching length in descending order
        self._contours = contours
        # Pose forest is a forest where nodes are puzzle piece indices and edges are transforms between them.
        # Edges points from children to parents, and the transformations stored in the edges align children to parents.
        self._pose_forest = nx.DiGraph()
        self._pose_forest.add_nodes_from(range(num_puzzle_pieces))
        self._current_edge_index = 0
        self._union_find = UnionFind(range(num_puzzle_pieces))

    def step(self) -> bool:
        """
        Add one new puzzle piece to the puzzle piece graph.
        :return: True if all the puzzle pieces have been processed, False otherwise.
        """
        # Skip if the two contours belong to the puzzle pieces that are already connected
        while True:
            if self._current_edge_index >= len(self._contour_graph_edges):
                return True
            matching_info: ContourMatchingResults = self._contour_graph_edges[
                self._current_edge_index
            ][2]
            source_puzzle_piece_id = self._contours[
                matching_info.source_contour_index
            ].puzzle_piece_id
            target_puzzle_piece_id = self._contours[
                matching_info.target_contour_index
            ].puzzle_piece_id
            if (
                self._union_find[source_puzzle_piece_id]
                != self._union_find[target_puzzle_piece_id]
            ):
                break
            self._current_edge_index += 1

        self._union_find.union(source_puzzle_piece_id, target_puzzle_piece_id)
        merge_pose_forest_nodes(
            self._pose_forest,
            target_puzzle_piece_id,
            source_puzzle_piece_id,
            matching_info.refined_matching_result.transform,
        )
        self._current_edge_index += 1
        return False

    def pose_forest(self) -> nx.DiGraph:
        return self._pose_forest
