"""Fuse the bundled Mamba scans without a runtime dependency on onnx.

The sparse protobuf schema preserves all unrecognized fields, including model
weights and type information. Only the twelve known Scan nodes are replaced.
"""
from functools import lru_cache

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


@lru_cache(maxsize=1)
def _model_type():
    schema = descriptor_pb2.FileDescriptorProto(name="typeseg_cpu.proto", package="typeseg_cpu")
    definitions = {
        "Model": [("graph", 7, "Graph", False), ("opset_import", 8, "Opset", True)],
        "Graph": [("node", 1, "Node", True), ("initializer", 5, "Tensor", True)],
        "Tensor": [("name", 8, "string", False)],
        "Node": [
            ("input", 1, "string", True), ("output", 2, "string", True),
            ("op_type", 4, "string", False), ("attribute", 5, "Attribute", True),
            ("domain", 7, "string", False),
        ],
        "Attribute": [
            ("name", 1, "string", False), ("i", 3, "int64", False),
            ("g", 6, "Graph", False), ("ints", 8, "int64", True),
            ("type", 20, "int32", False),
        ],
        "Opset": [("domain", 1, "string", False), ("version", 2, "int64", False)],
    }
    scalar_types = {"string": 9, "int64": 3, "int32": 5}
    for name, fields in definitions.items():
        message = schema.message_type.add(name=name)
        for field_name, number, kind, repeated in fields:
            field = message.field.add(
                name=field_name, number=number, label=3 if repeated else 1,
                type=scalar_types.get(kind, 11),
            )
            if kind not in scalar_types:
                field.type_name = ".typeseg_cpu." + kind
    pool = descriptor_pool.DescriptorPool()
    pool.Add(schema)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("typeseg_cpu.Model"))


def fuse_mamba_scans(content: bytes) -> bytes:
    model = _model_type()()
    model.ParseFromString(content)
    replaced = 0
    initial_states = set()
    for node in model.graph.node:
        if node.op_type != "Scan" or node.domain:
            continue
        attributes = {attribute.name: attribute for attribute in node.attribute}
        body_inputs = {name for inner in attributes["body"].g.node for name in inner.input}
        decay_names = [name for name in body_inputs if name.startswith("A_neg_")]
        skip_names = [name for name in body_inputs if name.startswith("D_")]
        directions = list(attributes["scan_input_directions"].ints)
        if (
            len(node.input) != 5 or len(node.output) != 2
            or len(decay_names) != 1 or len(skip_names) != 1
            or directions not in ([0, 0, 0, 0], [1, 1, 1, 1])
            or list(attributes["scan_output_directions"].ints) != [directions[0]]
        ):
            raise ValueError("Unsupported Mamba Scan graph")
        inputs = list(node.input[1:]) + decay_names + skip_names
        initial_states.add(node.input[0])
        output = node.output[1]
        node.Clear()
        node.input.extend(inputs)
        node.output.append(output)
        node.op_type = "SelectiveScan"
        node.domain = "typeseg.cpu"
        node.attribute.add(name="reverse", i=directions[0], type=2)
        replaced += 1
    if replaced != 12:
        raise ValueError("Expected twelve Mamba scans")
    initializers = [value for value in model.graph.initializer if value.name not in initial_states]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    model.opset_import.add(domain="typeseg.cpu", version=1)
    return model.SerializeToString()
