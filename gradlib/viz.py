from graphviz import Digraph

# draws computational graph
def draw(root_node, show_grad=False):
  nodes = set()
  edges = set()
  def dfs(node):
    if node not in nodes:
      nodes.add(node)
      for child in node.children:
        edges.add((child, node))
        dfs(child)
  dfs(root_node)
  dot = Digraph(format='svg', graph_attr={'rankdir':'TB'})
  for n in nodes:
    dot.node(name=str(id(n)), label=f"{n.value:.2f}, grad: {n.grad:.2f}" if show_grad else f"{n.value:.2f}", style="filled", fillcolor="#90ee90")
    if n.op:
      dot.node(name=str(id(n))+n.op, label=n.op, style="filled", fillcolor="#ff7f7f")
      dot.edge(str(id(n))+n.op, str(id(n)))
  for n1, n2 in edges:
    dot.edge(str(id(n1)), str(id(n2))+n2.op)
  return dot
  