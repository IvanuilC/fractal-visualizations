import matplotlib.pyplot as plt
import numpy as np
from numpy.random import choice
import turtle
from PIL import Image


def cantor(start, end, depth, current_depth=0, ax=None):
    if ax is None:
        ax = plt.gca()
    if current_depth > depth:
        return
    ax.hlines(current_depth, start, end, colors='k', lw=1)
    length = end - start
    left_end = start + length / 3
    right_start = end - length / 3
    cantor(start, left_end, depth, current_depth + 1, ax)
    cantor(right_start, end, depth, current_depth + 1, ax)


def cantor_visualisation():
    fig, ax = plt.subplots(figsize=(10, 4))
    cantor(0, 1, depth=5, ax=ax)
    ax.set_title("Канторово множество")
    ax.set_yticks([])
    plt.savefig("cantor_ifs.png", dpi=300)


def sierpinski_chaos(n_points=100000):
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    points = np.zeros((n_points, 2))
    point = np.array([0.5, 0.25])
    for i in range(n_points):
        v_idx = choice([0, 1, 2])
        point = (point + vertices[v_idx]) / 2
        points[i] = point
    return points


def sierpinski_visualisation():
    points = sierpinski_chaos()
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=0.1, c='k')
    plt.title("Треугольник Серпинского")
    plt.axis('equal')
    plt.axis('off')
    plt.savefig("sierpinski_stochastic.png", dpi=300)


def mandelbrot(c, max_iter=100):
    z = np.zeros_like(c, dtype=np.complex128)
    diverge = np.zeros(c.shape, dtype=bool)
    iterations = np.zeros(c.shape, dtype=int)
    for i in range(max_iter):
        z[~diverge] = z[~diverge] ** 2 + c[~diverge]
        diverge_new = (np.abs(z) > 4) & ~diverge
        iterations[diverge_new] = i
        diverge |= diverge_new
    return iterations


def mandelbrot_visualisation():
    xmin, xmax = -2.0, 0.5
    ymin, ymax = -1.25, 1.25
    width, height = 2000, 2000
    real = np.linspace(xmin, xmax, width)
    imag = np.linspace(ymin, ymax, height)
    C = real + imag[:, None] * 1j
    mandel = mandelbrot(C, max_iter=200)
    plt.figure(figsize=(12, 10))
    plt.imshow(mandel.T, cmap='hot', extent=[xmin, xmax, ymin, ymax])
    plt.title("Множество Мандельброта")
    plt.savefig("mandelbrot_vectorized.png", dpi=150)


def lsystem_dragon(depth=10, step=5):
    sequence = "FX"
    rules = {'X': 'X+YF+', 'Y': '-FX-Y'}
    for _ in range(depth):
        sequence = ''.join(rules.get(c, c) for c in sequence)
    screen = turtle.Screen()
    t = turtle.Turtle()
    t.speed(0)
    t.hideturtle()
    for cmd in sequence:
        if cmd == 'F':
            t.forward(step)
        elif cmd == '+':
            t.left(90)
        elif cmd == '-':
            t.right(90)
    screen.getcanvas().postscript(file="dragon.eps")
    screen.bye()


def convert_eps_to_png(eps_path, png_path, dpi=300):
    with Image.open(eps_path) as img:
        img.load(scale=10)
        img.save(png_path, 'PNG', dpi=(dpi, dpi))


def dragon_visualisation():
    lsystem_dragon(depth=12)
    convert_eps_to_png("dragon.eps", "dragon.png")


def main():
    cantor_visualisation()
    sierpinski_visualisation()
    mandelbrot_visualisation()
    dragon_visualisation()


main()
