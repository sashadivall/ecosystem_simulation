"""
File: alife.py
Description: A simple artificial life simulation.
"""
import random as rnd
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import argparse

# Define colors

colors = ['gray', 'green', 'blue', 'red']
cmap = ListedColormap(colors)

mpl.use('macosx')

SIZE = None  # x/y dimensions of the field
WRAP = True  # When moving beyond the border, do we wrap around to the other size
CYCLES = None
RABBIT_OFFSPRING = 2  # The number of offspring when a rabbit reproduces
FOX_OFFSPRING = 1  # The number of offspring when a fox reproduces
GRASS_RATE = None  # Probability of grass growing at any given location, e.g., 2%
INIT_RABBITS = None  # Number of starting rabbits
INIT_FOXES = None
SPEED = 1  # Number of generations per frame


class Animal:

    def __init__(self, type):
        self.type = type
        self.x = rnd.randrange(0, SIZE)
        self.y = rnd.randrange(0, SIZE)
        self.eaten = 0
        self.cycles = 0
        self.dead = False

    def reproduce(self):
        self.eaten = 0
        self.cycles = 0
        return copy.deepcopy(self)

    def eat(self, amount):
        self.eaten += amount

    def move(self):
        if self.type == 'rabbit':
            if WRAP:
                self.x = (self.x + rnd.choice([-1, 0, 1])) % SIZE
                self.y = (self.y + rnd.choice([-1, 0, 1])) % SIZE
            else:
                self.x = min(SIZE - 1, max(0, (self.x + rnd.choice([-1, 0, 1]))))
                self.y = min(SIZE - 1, max(0, (self.y + rnd.choice([-1, 0, 1]))))
        if self.type == 'fox':
            if WRAP:
                self.x = (self.x + rnd.choice([-2, -1, 0, 1, 2])) % SIZE
                self.y = (self.y + rnd.choice([-2, -1, 0, 1, 2])) % SIZE
            else:
                self.x = min(SIZE - 1, max(0, (self.x + rnd.choice([-2, -1, 0, 1, 2]))))
                self.y = min(SIZE - 1, max(0, (self.y + rnd.choice([-2, -1, 0, 1, 2]))))
            self.cycles += 1


class Field:
    """ A field is a patch of grass with 0 or more rabbits hopping around
    in search of grass """

    def __init__(self):
        self.rabbits = []
        self.foxes = []
        self.field = np.ones((SIZE, SIZE), dtype=int)

    def add_rabbit(self, rabbit):
        self.rabbits.append(rabbit)

    def add_fox(self, fox):
        self.foxes.append(fox)

    def move(self):
        for r in self.rabbits:
            r.move()
        for f in self.foxes:
            f.move()

    def eat(self):
        """ All rabbits try to eat grass at their current location """
        for r in self.rabbits:
            r.eat(self.field[r.x, r.y])
            self.field[r.x, r.y] = 0

        for f in self.foxes:
            x_fox = f.x
            y_fox = f.y
            for r in self.rabbits:
                x_rab = r.x
                y_rab = r.y
                if x_fox == x_rab and y_fox == y_rab:
                    f.eat(1)
                    r.dead = True
                else:
                    f.eat(0)

    def survive(self):
        """ Rabbits that have not eaten die. Otherwise, they live """
        self.rabbits = [r for r in self.rabbits if r.eaten > 0 and not r.dead]
        self.foxes = [f for f in self.foxes if f.eaten > 0 or f.cycles <= CYCLES]

    def reproduce(self):
        born_rabbits = []
        for r in self.rabbits:
            for _ in range(rnd.randint(1, RABBIT_OFFSPRING)):
                born_rabbits.append(r.reproduce())
        self.rabbits += born_rabbits
        born_foxes = []
        for f in [f for f in self.foxes if f.eaten > 0]:
            for _ in range(rnd.randint(1, FOX_OFFSPRING)):
                born_foxes.append(f.reproduce())
        self.foxes += born_foxes

    def grow(self):
        growloc = (np.random.rand(SIZE, SIZE) < GRASS_RATE) * 1
        self.field = np.maximum(self.field, growloc)

    def generation(self):
        """ Run one generation of rabbit actions """
        self.move()
        self.eat()
        self.survive()
        self.reproduce()
        self.grow()


def animate(i, field, im):

    for _ in range(SPEED):
        field.generation()

    grass_array = field.field
    rabbit_array = np.zeros((SIZE, SIZE), dtype=int)
    for r in field.rabbits:
        rabbit_array[r.x, r.y] = 2
    fox_array = np.zeros((SIZE, SIZE), dtype=int)
    for f in field.foxes:
        fox_array[f.x, f.y] = 3

    im.set_array(np.maximum(grass_array, fox_array, rabbit_array))
    plt.title(
        "Generation: " + str(i * SPEED) + " Rabbits: " + str(len(field.rabbits)) + " Foxes: " + str(len(field.foxes)))

    return im,


def main():
    parser = argparse.ArgumentParser(description='Artificial Life Simulation with Rabbits and Foxes')
    parser.add_argument('--grass_rate', type=float, default=0.1, help='Probability of Grass Growing')
    parser.add_argument('--fox_k', type=int, default=10, help='Number of Cycles a Fox Can Survive Without Eating')
    parser.add_argument('--field_size', type=int, default=100, help='X/Y Dimensions of the Field')
    parser.add_argument('--init_rabbits', type=int, default=1, help='Number of Initial Rabbits')
    parser.add_argument('--init_foxes', type=int, default=2, help='Number of Initial Foxes')
    args = parser.parse_args()

    global GRASS_RATE, CYCLES, SIZE, INIT_RABBITS, INIT_FOXES
    GRASS_RATE = args.grass_rate
    CYCLES = args.fox_k
    SIZE = args.field_size
    INIT_RABBITS = args.init_rabbits
    INIT_FOXES = args.init_foxes
    # Create the ecosystem
    field = Field()

    # Initialize with some rabbits
    for _ in range(INIT_RABBITS):
        field.add_rabbit(Animal(type='rabbit'))
    for _ in range(INIT_FOXES):
        field.add_fox(Animal(type='fox'))

    # Set up the image object
    array = np.ones(shape=(SIZE, SIZE), dtype=int)
    fig = plt.figure(figsize=(10, 10))
    im = plt.imshow(array, cmap=cmap, interpolation='hamming', aspect='auto', vmin=0, vmax=3)
    anim = animation.FuncAnimation(fig, animate, fargs=(field, im), frames=10 ** 100, interval=1, repeat=True)
    plt.show()

    generations = range(1001)
    rabbits_population = [len(field.rabbits)]
    foxes_population = [len(field.foxes)]
    grass_population = [np.sum(field.field)]
    for _ in range(1000):
        field.generation()
        rabbits_population.append(len(field.rabbits))
        foxes_population.append(len(field.foxes))
        grass_population.append(np.sum(field.field))

    plt.plot(generations, rabbits_population, label='Rabbits')
    plt.plot(generations, foxes_population, label='Foxes')
    plt.plot(generations, grass_population, label='Grass')
    plt.xlabel("Species")
    plt.ylabel("Population")
    plt.title(f"Rabbits vs. Grass vs. Foxes After {max(generations)} Generations, k = {CYCLES}")
    plt.legend()
    plt.savefig(f"viz/population_k{CYCLES}.png")


if __name__ == '__main__':
    main()
