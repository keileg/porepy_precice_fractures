#!python3

from classy_blocks import Mesh, Face, Extrude, Spline
from pathlib import Path
import numpy as np
import argparse

def meshFor(nx:int, amplitude:float, waves:int, aperture:float, shift:float):
    l, r = 0.0, 1.0
    t, b = aperture, 0.0

    xs = np.linspace(l, r, nx)
    ys = np.sin(waves*2*np.pi*xs)*amplitude
    ys_shifted = np.sin(waves*2*np.pi*xs + shift*2*np.pi)*amplitude

    top = [ [x, t+dy, 0.0 ] for x, dy in zip(xs, ys) ]
    bot = [ [x, b+dy, 0.0 ] for x, dy in zip(xs, ys_shifted) ]

    points = [
            bot[0],
            bot[-1],
            top[-1],
            top[0]
            ]

    edges = [
            Spline(bot[1:-2]),
            None,
            Spline(top[1:-2][::-1]),
            None,
            ]

    # create base face and extrude to a hex
    face = Face(points, edges)
    block = Extrude(face, amount=[0, 0, 0.1])

    # set patches
    block.set_patch("left", "inlet")
    block.set_patch("right", "outlet")
    block.set_patch("top", "frontAndBack")
    block.set_patch("bottom", "frontAndBack")
    block.set_patch("front", "lowerWall")  
    block.set_patch("back", "upperWall")

    mesh = Mesh()
    mesh.add(block)

    # update patches
    mesh.modify_patch("frontAndBack", "empty")
    mesh.modify_patch("upperWall", "wall")
    mesh.modify_patch("lowerWall", "wall")

    # flow direction
    block.chop(0, count=50)
    # top-bottom (wall)
    block.chop(1, total_expansion=10, count=30)
    block.chop(1, total_expansion=0.1, count=30)
    # back-front (empty)
    block.chop(2, count=1)

    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=100, type=int, help="points in x direction")
    parser.add_argument("-a", default=0.01, type=float, help="ampliture of wall wave")
    parser.add_argument("-w", default=10, type=int, help="full waves per wall")

    parser.add_argument("-b", default=0.1, type=float, help="aperture of fracture")
    parser.add_argument("-s", default=0.0, type=float, help="phase shift of one of the walls")

    args = parser.parse_args()

    meshFor(nx=args.n, amplitude=args.a, waves=args.w, aperture=args.b, shift=args.s).write(Path(__file__).parent / "system"/ "blockMeshDict")


if __name__ == "__main__":
    main()
