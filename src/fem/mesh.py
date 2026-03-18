from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


# Produce a mesh data structure that feeds FEM assembly cleanly.
@dataclass
class Mesh2D:
    nodes: np.ndarray  # shape (N, 2)
    elements: np.ndarray  # shape (T, 3)
    boundary_edges: np.ndarray  # shape (B, 2)
    boundary_tags: list[str]  # length B


def rectangle_mesh(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0) -> Mesh2D:
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)

    nodes = np.array([[x, y] for y in ys for x in xs], dtype=float)

    def node_id(i: int, j: int) -> int:
        return j * (nx + 1) + i

    elements = []
    for j in range(ny):
        for i in range(nx):
            n00 = node_id(i, j)
            n10 = node_id(i + 1, j)
            n01 = node_id(i, j + 1)
            n11 = node_id(i + 1, j + 1)

            elements.append([n00, n10, n11])
            elements.append([n00, n11, n01])

    elements = np.array(elements, dtype=int)

    boundary_edges = []
    boundary_tags = []

    # bottom
    for i in range(nx):
        boundary_edges.append([node_id(i, 0), node_id(i + 1, 0)])
        boundary_tags.append("bottom")

    # top
    for i in range(nx):
        boundary_edges.append([node_id(i, ny), node_id(i + 1, ny)])
        boundary_tags.append("top")

    # left
    for j in range(ny):
        boundary_edges.append([node_id(0, j), node_id(0, j + 1)])
        boundary_tags.append("left")

    # right
    for j in range(ny):
        boundary_edges.append([node_id(nx, j), node_id(nx, j + 1)])
        boundary_tags.append("right")

    return Mesh2D(
        nodes=nodes,
        elements=elements,
        boundary_edges=np.array(boundary_edges, dtype=int),
        boundary_tags=boundary_tags,
    )


def plot_mesh(
    mesh: Mesh2D,
    show_element_numbers: bool = True,
    show_element_color: bool = True,
    show_node_numbers: bool = False,
) -> None:
    fig, ax = plt.subplots()

    triang = mtri.Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.elements)

    ax.triplot(triang, linewidth=0.8)
    ax.plot(mesh.nodes[:, 0], mesh.nodes[:, 1], "o", markersize=3)

    # Plot element numbers
    if show_element_numbers:
        for k, elem in enumerate(mesh.elements):
            coords = mesh.nodes[elem]
            centroid = coords.mean(axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                str(k),
                color="black",
                fontsize=10,
                ha="center",
            )

    if show_element_color is True:
        colors = np.ones(len(mesh.elements))
        ax.tripcolor(triang, facecolors=colors, edgecolors="k", alpha=0.7)
    else:
        ax.triplot(triang)

    # Show Numbers
    if show_node_numbers:
        for k, (x, y) in enumerate(mesh.nodes):
            ax.text(x, y, str(k), fontsize=8)

    ax.set_aspect("equal")
    ax.set_title("Triangulated mesh")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()
