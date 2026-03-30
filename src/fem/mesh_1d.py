from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Mesh1D:
    nodes: np.ndarray          # shape (N, 1)
    elements: np.ndarray       # shape (E, 2)
    boundary_nodes: np.ndarray # shape (2,)
    boundary_tags: list[str]   # ["left", "right"]


def interval_mesh(nx: int, lx: float = 1.0) -> Mesh1D:
    xs = np.linspace(0.0, lx, nx + 1)
    # Column vector, each element is a coordinate for node
    nodes = xs[:, None]

    # Each element holds two numbers, i and i+1,
    elements = np.column_stack((
        np.arange(nx, dtype=int),
        np.arange(1, nx + 1, dtype=int),
    ))

    boundary_nodes = np.array([0, nx], dtype=int)
    boundary_tags = ["left", "right"]

    return Mesh1D(
        nodes=nodes,
        elements=elements,
        boundary_nodes=boundary_nodes,
        boundary_tags=boundary_tags,
    )


def boundary_nodes(mesh: Mesh1D, tags: set[str] | None = None) -> np.ndarray:
    chosen = []

    for node, tag in zip(mesh.boundary_nodes, mesh.boundary_tags):
        if tags is None or tag in tags:
            chosen.append(int(node))

    return np.array(chosen, dtype=int)


def plot_mesh(
    mesh: Mesh1D,
    show_element_numbers: bool = True,
    show_node_numbers: bool = False,
    save_path: str | None = None,
) -> None:
    fig, ax = plt.subplots()

    x = mesh.nodes[:, 0]
    y = np.zeros_like(x)

    for k, elem in enumerate(mesh.elements):
        xe = x[elem]
        ax.plot(xe, [0.0, 0.0], "k-", linewidth=1.2)
        if show_element_numbers:
            ax.text(xe.mean(), 0.02, str(k), ha="center")

    ax.plot(x, y, "o", color="C0")

    if show_node_numbers:
        for k, xk in enumerate(x):
            ax.text(xk, -0.03, str(k), ha="center")

    ax.set_yticks([])
    ax.set_xlabel("x")
    ax.set_title("1D mesh")
    ax.set_ylim(-0.08, 0.08)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
