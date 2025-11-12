
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as PltCircle

# --- Geometry helpers (adapted) ---
class Circle:
    def __init__(self, radius):
        self.radius = radius
        self.x = 0.0
        self.y = 0.0
        self.placed = False

    def set_position(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.placed = True

    def distance_from_origin(self):
        return math.hypot(self.x, self.y)

    def containing_radius(self):
        return self.distance_from_origin() + self.radius

    def distance_to(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def overlaps(self, other):
        distance = self.distance_to(other)
        return distance < (self.radius + other.radius - 1e-6)

# --- Bunch: manages many circles and open-point calculations ---
class Bunch:
    def __init__(self, radii):
        self.circles = [Circle(r) for r in radii]
        self.radii = list(radii)

    def reset(self):
        for c in self.circles:
            c.x = 0.0
            c.y = 0.0
            c.placed = False

    def get_containing_radius(self):
        placed = [c for c in self.circles if c.placed]
        if not placed:
            return 0.0
        return max(c.containing_radius() for c in placed)

    def find_open_points(self, new_circle):
        open_points = []
        placed = [c for c in self.circles if c.placed]

        if len(placed) == 0:
            return [(0.0, 0.0, 0.0)]

        if len(placed) == 1:
            c1 = placed[0]
            distance = c1.radius + new_circle.radius
            for angle in np.linspace(0, 2*math.pi, 36, endpoint=False):
                x = c1.x + distance * math.cos(angle)
                y = c1.y + distance * math.sin(angle)
                dist_from_origin = math.hypot(x, y)
                open_points.append((x, y, dist_from_origin))
            return open_points

        # tangent positions with pairs
        for i in range(len(placed)):
            for j in range(i+1, len(placed)):
                c1 = placed[i]
                c2 = placed[j]
                positions = self.find_tangent_positions(c1, c2, new_circle)
                for x, y in positions:
                    temp = Circle(new_circle.radius)
                    temp.set_position(x, y)
                    valid = True
                    for other in placed:
                        if other is c1 or other is c2:
                            continue
                        if temp.overlaps(other):
                            valid = False
                            break
                    if valid:
                        open_points.append((x, y, math.hypot(x, y)))
        return open_points

    def find_tangent_positions(self, c1, c2, new_circle):
        r1 = c1.radius + new_circle.radius
        r2 = c2.radius + new_circle.radius
        d = c1.distance_to(c2)
        if d == 0 or d > r1 + r2 or d < abs(r1 - r2):
            return []
        a = (r1*r1 - r2*r2 + d*d) / (2*d)
        h = math.sqrt(max(0.0, r1*r1 - a*a))
        px = c1.x + a * (c2.x - c1.x) / d
        py = c1.y + a * (c2.y - c1.y) / d
        positions = []
        if h > 1e-6:
            rx = (h * (c2.y - c1.y) / d)
            ry = (h * (c2.x - c1.x) / d)
            positions.append((px + rx, py - ry))
            positions.append((px - rx, py + ry))
        else:
            positions.append((px, py))
        return positions

    def ordered_place(self):
        self.reset()
        for c in self.circles:
            open_points = self.find_open_points(c)
            if open_points:
                best = min(open_points, key=lambda p: p[2])
                c.set_position(best[0], best[1])

    def random_place(self):
        self.reset()
        shuffled = self.circles[:]
        random.shuffle(shuffled)
        for c in shuffled:
            open_points = self.find_open_points(c)
            if open_points:
                best = min(open_points, key=lambda p: p[2])
                c.set_position(best[0], best[1])
            else:
                c.set_position(0.0, 0.0)

    def greedy_place(self):
        self.reset()
        sorted_circs = sorted(self.circles, key=lambda c: c.radius, reverse=True)
        for c in sorted_circs:
            open_points = self.find_open_points(c)
            if open_points:
                best = min(open_points, key=lambda p: p[2])
                c.set_position(best[0], best[1])
            else:
                c.set_position(0.0, 0.0)

    def draw(self, title="Circle Placement", save_path=None, containing_circle=True):
        fig, ax = plt.subplots(figsize=(10, 10))
        containing_radius = self.get_containing_radius()
        if containing_circle:
            cc = PltCircle((0, 0), containing_radius, fill=False, edgecolor='#F4BA02', linewidth=2, linestyle='--')
            ax.add_patch(cc)
        for c in self.circles:
            if c.placed:
                patch = PltCircle((c.x, c.y), c.radius, fill=False, edgecolor='#99D9DD', linewidth=2)
                ax.add_patch(patch)
                ax.plot(c.x, c.y, 'o', color='#99D9DD')
                ax.text(c.x, c.y, f"{int(c.radius)}", ha='center', va='center', color='#F7F8F9', fontsize=9)
        ax.plot(0, 0, 'x', color='#F4BA02')
        ax.set_aspect('equal')
        margin = max(1.0, containing_radius * 0.15)
        ax.set_xlim(-containing_radius - margin, containing_radius + margin)
        ax.set_ylim(-containing_radius - margin, containing_radius + margin)
        ax.grid(True, alpha=0.3)
        plt.title(title)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        plt.close(fig)

# --- Evolutionary Algorithm (searches permutations) ---
class PackingEA:
    def __init__(self, radii, weights, width, depth, weight_limit, pop_size=40, generations=200):
        self.radii = list(radii)
        self.weights = list(weights)
        assert len(self.radii) == len(self.weights)
        self.N = len(self.radii)
        self.width = float(width)
        self.depth = float(depth)
        self.weight_limit = float(weight_limit)
        self.pop_size = int(pop_size)
        self.generations = int(generations)

    def decode_and_place(self, order):
        b = Bunch(self.radii)
        b.reset()
        # We assume container center at (0,0) and bounds [-width/2, width/2] and [-depth/2, depth/2]
        for idx in order:
            circle = b.circles[idx]
            open_points = b.find_open_points(circle)
            valid_points = []
            for (x, y, dist) in open_points:
                if (x - circle.radius) >= -self.width/2 - 1e-8 and (x + circle.radius) <= self.width/2 + 1e-8 and \
                   (y - circle.radius) >= -self.depth/2 - 1e-8 and (y + circle.radius) <= self.depth/2 + 1e-8:
                    valid_points.append((x, y, dist))
            if valid_points:
                best = min(valid_points, key=lambda p: p[2])
                circle.set_position(best[0], best[1])
            else:
                # fallback attempts
                placed = [c for c in b.circles if c.placed]
                placed_ok = False
                if len(placed) == 0:
                    y0 = -self.depth/2 + circle.radius
                    circle.set_position(0.0, y0)
                    placed_ok = True
                else:
                    for pc in placed:
                        d = pc.radius + circle.radius
                        for angle in np.linspace(0, 2*math.pi, 24, endpoint=False):
                            x = pc.x + d * math.cos(angle)
                            y = pc.y + d * math.sin(angle)
                            if (x - circle.radius) < -self.width/2 or (x + circle.radius) > self.width/2:
                                continue
                            if (y - circle.radius) < -self.depth/2 or (y + circle.radius) > self.depth/2:
                                continue
                            temp = Circle(circle.radius)
                            temp.set_position(x, y)
                            ok = True
                            for other in placed:
                                if temp.overlaps(other):
                                    ok = False
                                    break
                            if ok:
                                circle.set_position(x, y)
                                placed_ok = True
                                break
                        if placed_ok:
                            break
                if not placed_ok:
                    circle.set_position(0.0, 0.0)
        return b

    def compute_fitness(self, b):
        overlap = 0.0
        placed = [c for c in b.circles if c.placed]
        for i in range(len(placed)):
            for j in range(i+1, len(placed)):
                d = placed[i].distance_to(placed[j])
                diff = (placed[i].radius + placed[j].radius) - d
                if diff > 0:
                    overlap += diff
        oob = 0.0
        for c in placed:
            left = -self.width/2 + c.radius
            right = self.width/2 - c.radius
            bottom = -self.depth/2 + c.radius
            top = self.depth/2 - c.radius
            if c.x < left:
                oob += (left - c.x)
            if c.x > right:
                oob += (c.x - right)
            if c.y < bottom:
                oob += (bottom - c.y)
            if c.y > top:
                oob += (c.y - top)
        total_weight = 0.0
        com_x = 0.0
        com_y = 0.0
        for idx, c in enumerate(b.circles):
            if c.placed:
                w = self.weights[idx]
                total_weight += w
                com_x += w * c.x
                com_y += w * c.y
        if total_weight > 0:
            com_x /= total_weight
            com_y /= total_weight
        else:
            com_x = 0.0
            com_y = 0.0
        half_allowed_w = 0.6 * self.width / 2.0
        half_allowed_d = 0.6 * self.depth / 2.0
        com_pen = 0.0
        if abs(com_x) > half_allowed_w:
            com_pen += abs(abs(com_x) - half_allowed_w)
        if abs(com_y) > half_allowed_d:
            com_pen += abs(abs(com_y) - half_allowed_d)
        weight_pen = 0.0
        if total_weight > self.weight_limit:
            weight_pen = (total_weight - self.weight_limit)
        fitness = 1000.0 * overlap + 500.0 * oob + 50.0 * com_pen + 100.0 * weight_pen
        return fitness, {
            'overlap': overlap,
            'oob': oob,
            'com_pen': com_pen,
            'weight_pen': weight_pen,
            'total_weight': total_weight,
            'com': (com_x, com_y)
        }

    def random_permutation(self):
        p = list(range(self.N))
        random.shuffle(p)
        return p

    def order_crossover(self, p1, p2):
        a, b = sorted(random.sample(range(self.N), 2))
        child = [-1] * self.N
        child[a:b+1] = p1[a:b+1]
        cur = 0
        for gene in p2:
            if gene in child:
                continue
            while child[cur] != -1:
                cur += 1
            child[cur] = gene
        return child

    def swap_mutation(self, p, pm=0.2):
        q = p[:]
        if random.random() < pm:
            i, j = random.sample(range(self.N), 2)
            q[i], q[j] = q[j], q[i]
        return q

    def tournament_select(self, population, fitnesses, k=3):
        i = random.randrange(len(population))
        best = i
        for _ in range(k-1):
            j = random.randrange(len(population))
            if fitnesses[j] < fitnesses[best]:
                best = j
        return population[best]

    def local_search_improve(self, perm, evaluations=50):
        best_perm = perm[:]
        best_b = self.decode_and_place(best_perm)
        best_f, _ = self.compute_fitness(best_b)
        for _ in range(evaluations):
            i, j = random.sample(range(self.N), 2)
            cand = best_perm[:]
            cand[i], cand[j] = cand[j], cand[i]
            b = self.decode_and_place(cand)
            f, _ = self.compute_fitness(b)
            if f < best_f:
                best_f = f
                best_perm = cand
                best_b = b
        return best_perm, best_b, best_f

    def run(self, verbose=True):
        pop = [self.random_permutation() for _ in range(self.pop_size)]
        if self.pop_size >= 1:
            pop[0] = list(range(self.N))
        if self.pop_size >= 2:
            pop[1] = sorted(range(self.N), key=lambda i: -self.radii[i])
        fitnesses = []
        decoded = []
        for p in pop:
            b = self.decode_and_place(p)
            f, _ = self.compute_fitness(b)
            fitnesses.append(f)
            decoded.append(b)
        best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
        best_perm = pop[best_idx]
        best_b = decoded[best_idx]
        best_f = fitnesses[best_idx]
        if verbose:
            print(f"Initial best fitness: {best_f:.3f}")
        for gen in range(self.generations):
            new_pop = []
            new_fit = []
            new_decoded = []
            while len(new_pop) < self.pop_size:
                parent1 = self.tournament_select(pop, fitnesses)
                parent2 = self.tournament_select(pop, fitnesses)
                child = self.order_crossover(parent1, parent2)
                child = self.swap_mutation(child, pm=0.3)
                if random.random() < 0.3:
                    child, child_b, child_f = self.local_search_improve(child, evaluations=20)
                else:
                    child_b = self.decode_and_place(child)
                    child_f, _ = self.compute_fitness(child_b)
                new_pop.append(child)
                new_fit.append(child_f)
                new_decoded.append(child_b)
            pop = new_pop
            fitnesses = new_fit
            decoded = new_decoded
            gen_best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
            if fitnesses[gen_best_idx] < best_f:
                best_f = fitnesses[gen_best_idx]
                best_perm = pop[gen_best_idx]
                best_b = decoded[gen_best_idx]
            if verbose and (gen % max(1, self.generations//10) == 0):
                print(f"Gen {gen}: best fitness {best_f:.3f}")
            if best_f == 0.0:
                if verbose:
                    print(f"Found perfect solution at generation {gen}")
                break
        return best_perm, best_b, best_f

# --- Runner: example usage ---
if __name__ == '__main__':
    # Example problem (from notebook brief)
    radii = [10, 34, 10, 55, 30, 14, 70, 14]
    # Example weights (arbitrary for demo) -- same length as radii
    weights = [50, 120, 45, 200, 100, 40, 260, 35]
    width = 20.0
    depth = 15.0
    weight_limit = 2000.0

    ea = PackingEA(radii, weights, width, depth, weight_limit, pop_size=20, generations=100)
    best_perm, best_b, best_f = ea.run(verbose=True)
    print('Best fitness:', best_f)
    fit, details = ea.compute_fitness(best_b)
    print('Details:', details)
    # Save visualization
    best_b.draw(title=f"Best packing (fitness={best_f:.3f})", save_path='best_packing.png')
    print('Permutation (loading order):', best_perm)
