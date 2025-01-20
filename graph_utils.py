import functools

import tensorflow as tf
import tensorflow_gnn as tfgnn

from data_utils import blocks_from_orca, serv_arg_name


def graph_from_orca(out_file: str,
                    overlap_thresh: float,
                    target_features: dict,
                    dummy: bool = False):

    diagonal_densities,               \
        off_diagonal_densities,       \
        off_diagonal_overlaps,        \
        adjacency_atom2link_sources,  \
        adjacency_atom2link_targets,  \
        adjacency_link2atom_sources,  \
        adjacency_link2atom_targets,  \
        atoms, natoms, nlinks, nbas = \
        blocks_from_orca(out_file, overlap_thresh, dummy=dummy)

    graph = tfgnn.GraphTensor.from_pieces(
        context=tfgnn.Context.from_fields(
            features={target_feature: tf.constant([target_value])
                      for target_feature, target_value
                      in target_features.items()}),
        node_sets={
            "atom": tfgnn.NodeSet.from_fields(
                sizes=tf.constant([natoms]),
                features={"density": tf.constant(diagonal_densities),
                          "nuc_charge": tf.constant(atoms)}),
            "link": tfgnn.NodeSet.from_fields(
                sizes=tf.constant([nlinks]),
                features={"density": tf.constant(off_diagonal_densities)})},
        edge_sets={
            "atom2link": tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([nlinks]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("atom", tf.constant(adjacency_atom2link_sources)),
                    target=("link", tf.constant(adjacency_atom2link_targets))),
                features={"overlap": tf.constant(off_diagonal_overlaps)}),
            "link2atom": tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([nlinks]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("link", tf.constant(adjacency_link2atom_sources)),
                    target=("atom", tf.constant(adjacency_link2atom_targets))),
                features={"overlap": tf.constant(off_diagonal_overlaps)})})

    return graph, nbas


###############################################################################
# TF-GNN data manipulation utilities
###############################################################################
def read_schema(schema_path: str):
    schema = tfgnn.read_schema(schema_path)

    return tfgnn.create_graph_spec_from_schema_pb(schema)


def decode_graph(record_bytes, graph_spec=None, targets=None):
    assert graph_spec is not None, "No graph specification provided"
    graph = tfgnn.parse_single_example(graph_spec, record_bytes, validate=True)
    if targets:
        return graph, {target: graph.context[target] for target in targets}
    else:
        return graph


def get_decode_fn(graph_spec, targets=None):
    return functools.partial(decode_graph,
                             graph_spec=graph_spec,
                             targets=targets)





class GraphToTensorsLayer(tf.keras.layers.Layer):
    def __init__(self, graph_schema: tfgnn.GraphSchema):
        super().__init__()
        self.schema = {
            "context": {"features": {
                key: graph_schema.context_spec._data_spec["features"][key]
                for key in sorted(
                    graph_schema.context_spec._data_spec["features"])},
                "sizes": graph_schema.context_spec._data_spec["sizes"]},
            "edge_sets": {
                key: {"adjacency": {
                    '#index.0': tf.TensorSpec(shape=(None,), dtype=tf.int32,
                                              name=None),
                    '#index.1': tf.TensorSpec(shape=(None,), dtype=tf.int32,
                                              name=None)},
                      "features": {edge_key:
                                       graph_schema.edge_sets_spec[
                                           key]._data_spec[
                                           "features"][edge_key]
                                   for edge_key in sorted(
                              graph_schema.edge_sets_spec[key]._data_spec[
                                  "features"])},
                      "sizes": graph_schema.edge_sets_spec[key]._data_spec[
                          "sizes"]}
                for key in sorted(graph_schema.edge_sets_spec)
            },
            "node_sets": {
                key: {"features": {node_key:
                                       graph_schema.node_sets_spec[
                                           key]._data_spec[
                                           "features"][node_key]
                                   for node_key in sorted(
                        graph_schema.node_sets_spec[
                            key]._data_spec["features"])},
                      "sizes": graph_schema.node_sets_spec[key]._data_spec[
                          "sizes"]}
                for key in sorted(graph_schema.node_sets_spec)
            }
        }

    def call(self, graph: tfgnn.GraphTensor):

        output_tensors = {}
        idx = serv_arg_name()

        # Map context features
        for feature_key in sorted(self.schema["context"]["features"]):
            output_tensors[next(idx)] = \
                graph.context.features[feature_key]
        output_tensors[next(idx)] = graph.context.sizes

        # Map edges features
        for edge_key in sorted(self.schema["edge_sets"]):
            output_tensors[next(idx)] = \
                graph.edge_sets[edge_key].adjacency.source
            output_tensors[next(idx)] = \
                graph.edge_sets[edge_key].adjacency.target
            for feature_key in sorted(
                    self.schema["edge_sets"][edge_key]["features"]):
                output_tensors[next(idx)] = \
                    graph.edge_sets[edge_key].features[feature_key]
            output_tensors[next(idx)] = graph.edge_sets[edge_key].sizes

        # Map nodes features
        for node_key in sorted(self.schema["node_sets"]):
            for feature_key in sorted(
                    self.schema["node_sets"][node_key]["features"]):
                output_tensors[next(idx)] = \
                    graph.node_sets[node_key].features[feature_key]
            output_tensors[next(idx)] = graph.node_sets[node_key].sizes

        return output_tensors


def graph2rest(graph: tfgnn.GraphTensor, mapping_layer: tf.keras.layers.Layer):
    mapped_graph = mapping_layer(graph)
    mapped_graph = {feature_name: feature_tensor.numpy().tolist()
                    for feature_name, feature_tensor in mapped_graph.items()}
    return mapped_graph
