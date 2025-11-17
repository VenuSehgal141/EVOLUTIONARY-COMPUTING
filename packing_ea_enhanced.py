r"""
Enhanced Circle Packing Evolutionary Algorithm
================================================
Implements recommendations:
1. Continuous local relaxation (gradient-based position refinement)
2. Enhanced candidate generation with adaptive angle sampling
3. Random restarts capability
4. Rectangular bounds and weight/COM constraints
5. Hybrid constructive + evolutionary approach

Author: Evolved Implementation
Date: 2025
"""

import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as PltCircle

# --- Geometry helpers ---
class Circle:
    """Represents a circle with position and geometric operations"""
    def __init__(self, radius):
        self.radius = radius
        self.x = 0.0
        self.y = 0.0
        self.placed = False
        self.vx = 0.0  # velocity for relaxation
        self.vy = 0.0

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
        min_dist = self.radius + other.radius - 1e-6
        return distance < min_dist

    def overlap_amount(self, other):
        """Returns positive amount of overlap, 0 if no overlap"""
        distance = self.distance_to(other)
        return max(0.0, self.radius + other.radius - distance)

    def is_within_bounds(self, width, depth):
        """Check if circle is within rectangular bounds"""
        return (abs(self.x) + self.radius <= width/2 and 
                abs(self.y) + self.radius <= depth/2)

    def copy(self):
        """Create an independent copy"""
        c = Circle(self.radius)
        c.set_position(self.x, self.y)
        c.vx = self.vx
        c.vy = self.vy
        return c


# --- Bunch: manages many circles and placement methods ---
class Bunch:
    """Manages collection of circles and placement algorithms"""
    def __init__(self, radii):
        self.circles = [Circle(r) for r in radii]
        self.radii = list(radii)

    def reset(self):
        """Reset all circles to unplaced state"""
        for c in self.circles:
            c.x = 0.0
            c.y = 0.0
            c.placed = False
            c.vx = 0.0
            c.vy = 0.0

    def copy(self):
        """Create deep copy of bunch"""
        b = Bunch(self.radii)
        for i, c in enumerate(self.circles):
            b.circles[i] = c.copy()
        return b

    def get_containing_radius(self):
        """Get radius of smallest circle containing all placed circles"""
        placed = [c for c in self.circles if c.placed]
        if not placed:
            return 0.0
        return max(c.containing_radius() for c in placed)

    def find_open_points(self, new_circle, num_angles=36):
        """
        Find candidate positions for placing a new circle.
        Enhanced with adaptive angle sampling.
        """
        open_points = []
        placed = [c for c in self.circles if c.placed]

        if len(placed) == 0:
            return [(0.0, 0.0, 0.0)]

        if len(placed) == 1:
            c1 = placed[0]
            distance = c1.radius + new_circle.radius
            for angle in np.linspace(0, 2*math.pi, num_angles, endpoint=False):
                x = c1.x + distance * math.cos(angle)
                y = c1.y + distance * math.sin(angle)
                dist_from_origin = math.hypot(x, y)
                open_points.append((x, y, dist_from_origin))
            return open_points

        # Find tangent positions with all pairs
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
        
        # Add adaptive angle sampling around placed circles
        for c1 in placed:
            # Adaptive angles based on space availability
            num_adaptive = max(12, int(num_angles * 0.3))
            for angle in np.linspace(0, 2*math.pi, num_adaptive, endpoint=False):
                distance = c1.radius + new_circle.radius
                x = c1.x + distance * math.cos(angle)
                y = c1.y + distance * math.sin(angle)
                temp = Circle(new_circle.radius)
                temp.set_position(x, y)
                valid = all(not temp.overlaps(other) for other in placed if other is not c1)
                if valid:
                    open_points.append((x, y, math.hypot(x, y)))

        return open_points

    def find_tangent_positions(self, c1, c2, new_circle):
        """Find positions tangent to two circles"""
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

    def local_relaxation(self, max_iterations=50, damping=0.8):
        """
        Continuous local relaxation using force-based approach.
        Reduces overlaps by pushing circles apart while preserving tangencies.
        """
        placed = [c for c in self.circles if c.placed]
        
        for iteration in range(max_iterations):
            max_movement = 0.0
            
            for i, c in enumerate(placed):
                fx, fy = 0.0, 0.0
                
                # Repulsion forces from overlapping circles
                for j, other in enumerate(placed):
                    if i == j:
                        continue
                    overlap = c.overlap_amount(other)
                    if overlap > 1e-6:
                        dx = c.x - other.x
                        dy = c.y - other.y
                        dist = math.hypot(dx, dy) + 1e-9
                        # Repulsive force proportional to overlap
                        force = overlap * 2.0
                        fx += force * dx / dist
                        fy += force * dy / dist
                
                # Small damping to center (gentle centering bias)
                center_bias = 0.1
                fx -= center_bias * c.x
                fy -= center_bias * c.y
                
                # Update velocity with damping
                c.vx = damping * c.vx + fx * 0.05
                c.vy = damping * c.vy + fy * 0.05
                
                # Update position
                old_x, old_y = c.x, c.y
                c.x += c.vx
                c.y += c.vy
                
                movement = math.hypot(c.x - old_x, c.y - old_y)
                max_movement = max(max_movement, movement)
            
            # Convergence check
            if max_movement < 0.01:
                break
    
    def validate_placement(self, width=None, depth=None):
        """Check validity of current placement"""
        placed = [c for c in self.circles if c.placed]
        
        # Check overlaps
        for i in range(len(placed)):
            for j in range(i+1, len(placed)):
                if placed[i].overlaps(placed[j]):
                    return False
        
        # Check bounds if specified
        if width is not None and depth is not None:
            for c in placed:
                if not c.is_within_bounds(width, depth):
                    return False
        
        return True

    def ordered_place(self):
        """Place circles in order, always choosing position closest to center"""
        self.reset()
        for c in self.circles:
            open_points = self.find_open_points(c)
            if open_points:
                best = min(open_points, key=lambda p: p[2])
                c.set_position(best[0], best[1])

    def random_place(self):
        """Place circles in random order"""
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
        """Place larger circles first (greedy by radius)"""
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
        """Visualize circle placement"""
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
        ax.plot(0, 0, 'x', color='#F4BA02', linewidth=3)
        ax.set_aspect('equal')
        margin = max(1.0, containing_radius * 0.15)
        ax.set_xlim(-containing_radius - margin, containing_radius + margin)
        ax.set_ylim(-containing_radius - margin, containing_radius + margin)
        ax.grid(True, alpha=0.3)
        plt.title(title, fontsize=14, fontweight='bold')
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")
        else:
            plt.show()
        plt.close(fig)


# --- Enhanced Evolutionary Algorithm ---
class PackingEA:
    """
    Evolutionary Algorithm with hybrid constructive + optimization approach.
    Includes random restarts, local relaxation, and multi-objective fitness.
    """
    def __init__(self, radii, weights, width, depth, weight_limit, 
                 pop_size=40, generations=200, random_restarts=3):
        self.radii = list(radii)
        self.weights = list(weights)
        assert len(self.radii) == len(self.weights)
        self.N = len(self.radii)
        self.width = float(width)
        self.depth = float(depth)
        self.weight_limit = float(weight_limit)
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.random_restarts = int(random_restarts)
        self.best_overall = None
        self.best_overall_f = float('inf')

    def decode_and_place(self, order, use_relaxation=True):
        """
        Decode permutation to placement.
        Uses enhanced candidate generation and optional local relaxation.
        """
        b = Bunch(self.radii)
        b.reset()
        
        for idx in order:
            circle = b.circles[idx]
            open_points = b.find_open_points(circle, num_angles=48)
            valid_points = []
            
            for (x, y, dist) in open_points:
                if (x - circle.radius) >= -self.width/2 - 1e-8 and \
                   (x + circle.radius) <= self.width/2 + 1e-8 and \
                   (y - circle.radius) >= -self.depth/2 - 1e-8 and \
                   (y + circle.radius) <= self.depth/2 + 1e-8:
                    valid_points.append((x, y, dist))
            
            if valid_points:
                best = min(valid_points, key=lambda p: p[2])
                circle.set_position(best[0], best[1])
            else:
                # Fallback placement strategy
                placed = [c for c in b.circles if c.placed]
                placed_ok = False
                
                if len(placed) == 0:
                    y0 = -self.depth/2 + circle.radius
                    circle.set_position(0.0, y0)
                    placed_ok = True
                else:
                    for pc in placed:
                        d = pc.radius + circle.radius
                        for angle in np.linspace(0, 2*math.pi, 36, endpoint=False):
                            x = pc.x + d * math.cos(angle)
                            y = pc.y + d * math.sin(angle)
                            if (x - circle.radius) < -self.width/2 or \
                               (x + circle.radius) > self.width/2:
                                continue
                            if (y - circle.radius) < -self.depth/2 or \
                               (y + circle.radius) > self.depth/2:
                                continue
                            temp = Circle(circle.radius)
                            temp.set_position(x, y)
                            ok = all(not temp.overlaps(other) for other in placed)
                            if ok:
                                circle.set_position(x, y)
                                placed_ok = True
                                break
                        if placed_ok:
                            break
                
                if not placed_ok:
                    circle.set_position(0.0, 0.0)
        
        # Apply local relaxation if enabled
        if use_relaxation:
            b.local_relaxation(max_iterations=50, damping=0.7)
        
        return b

    def compute_fitness(self, b):
        """
        Multi-objective fitness function:
        - Minimize overlap
        - Minimize out-of-bounds
        - Minimize center-of-mass deviation
        - Minimize excess weight
        """
        placed = [c for c in b.circles if c.placed]
        
        # Overlap penalty
        overlap = 0.0
        for i in range(len(placed)):
            for j in range(i+1, len(placed)):
                d = placed[i].distance_to(placed[j])
                diff = (placed[i].radius + placed[j].radius) - d
                if diff > 0:
                    overlap += diff * diff  # Quadratic penalty
        
        # Out-of-bounds penalty
        oob = 0.0
        for c in placed:
            left = -self.width/2 + c.radius
            right = self.width/2 - c.radius
            bottom = -self.depth/2 + c.radius
            top = self.depth/2 - c.radius
            if c.x < left:
                oob += (left - c.x) ** 2
            if c.x > right:
                oob += (c.x - right) ** 2
            if c.y < bottom:
                oob += (bottom - c.y) ** 2
            if c.y > top:
                oob += (c.y - top) ** 2
        
        # Center-of-mass and weight constraints
        total_weight = sum(self.weights[i] for i, c in enumerate(b.circles) if c.placed)
        com_x = com_y = 0.0
        
        if total_weight > 0:
            for i, c in enumerate(b.circles):
                if c.placed:
                    com_x += self.weights[i] * c.x
                    com_y += self.weights[i] * c.y
            com_x /= total_weight
            com_y /= total_weight
        
        half_allowed_w = 0.6 * self.width / 2.0
        half_allowed_d = 0.6 * self.depth / 2.0
        com_pen = 0.0
        if abs(com_x) > half_allowed_w:
            com_pen += (abs(com_x) - half_allowed_w) ** 2
        if abs(com_y) > half_allowed_d:
            com_pen += (abs(com_y) - half_allowed_d) ** 2
        
        weight_pen = max(0.0, total_weight - self.weight_limit) ** 2
        
        # Combined fitness
        fitness = 1000.0 * overlap + 500.0 * oob + 50.0 * com_pen + 100.0 * weight_pen
        
        return fitness, {
            'overlap': overlap,
            'oob': oob,
            'com_pen': com_pen,
            'weight_pen': weight_pen,
            'total_weight': total_weight,
            'com': (com_x, com_y),
            'num_placed': len(placed)
        }

    def random_permutation(self):
        """Generate random permutation"""
        p = list(range(self.N))
        random.shuffle(p)
        return p

    def order_crossover(self, p1, p2):
        """Order crossover (OX) operator"""
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
        """Swap mutation"""
        q = p[:]
        if random.random() < pm:
            i, j = random.sample(range(self.N), 2)
            q[i], q[j] = q[j], q[i]
        return q

    def tournament_select(self, population, fitnesses, k=3):
        """Tournament selection"""
        i = random.randrange(len(population))
        best = i
        for _ in range(k-1):
            j = random.randrange(len(population))
            if fitnesses[j] < fitnesses[best]:
                best = j
        return population[best]

    def local_search_improve(self, perm, evaluations=50):
        """Local search via neighboring permutations"""
        best_perm = perm[:]
        best_b = self.decode_and_place(best_perm, use_relaxation=True)
        best_f, _ = self.compute_fitness(best_b)
        
        for _ in range(evaluations):
            i, j = random.sample(range(self.N), 2)
            cand = best_perm[:]
            cand[i], cand[j] = cand[j], cand[i]
            b = self.decode_and_place(cand, use_relaxation=True)
            f, _ = self.compute_fitness(b)
            if f < best_f:
                best_f = f
                best_perm = cand
                best_b = b
        
        return best_perm, best_b, best_f

    def run_single_generation_sequence(self, verbose=True):
        """Run single EA sequence"""
        pop = [self.random_permutation() for _ in range(self.pop_size)]
        
        # Seed with good heuristics
        if self.pop_size >= 1:
            pop[0] = list(range(self.N))
        if self.pop_size >= 2:
            pop[1] = sorted(range(self.N), key=lambda i: -self.radii[i])
        if self.pop_size >= 3:
            pop[2] = sorted(range(self.N), key=lambda i: self.radii[i])
        
        # Evaluate initial population
        fitnesses = []
        decoded = []
        for p in pop:
            b = self.decode_and_place(p, use_relaxation=True)
            f, _ = self.compute_fitness(b)
            fitnesses.append(f)
            decoded.append(b)
        
        best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
        best_perm = pop[best_idx]
        best_b = decoded[best_idx]
        best_f = fitnesses[best_idx]
        
        if verbose:
            print(f"  Initial best fitness: {best_f:.2f}")
        
        # Evolution loop
        for gen in range(self.generations):
            new_pop = []
            new_fit = []
            new_decoded = []
            
            while len(new_pop) < self.pop_size:
                parent1 = self.tournament_select(pop, fitnesses, k=3)
                parent2 = self.tournament_select(pop, fitnesses, k=3)
                child = self.order_crossover(parent1, parent2)
                child = self.swap_mutation(child, pm=0.3)
                
                if random.random() < 0.4:
                    child, child_b, child_f = self.local_search_improve(child, evaluations=30)
                else:
                    child_b = self.decode_and_place(child, use_relaxation=True)
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
                print(f"  Gen {gen:3d}: best = {best_f:.2f}")
            
            if best_f < 1.0:  # Near-optimal
                if verbose:
                    print(f"  Converged at generation {gen}")
                break
        
        return best_perm, best_b, best_f

    def run(self, verbose=True):
        """
        Run with random restarts for better exploration.
        """
        print(f"\n{'='*60}")
        print(f"Packing EA with {self.random_restarts} random restart(s)")
        print(f"Population size: {self.pop_size}, Generations: {self.generations}")
        print(f"{'='*60}\n")
        
        for restart in range(self.random_restarts):
            if verbose:
                print(f"Restart {restart + 1}/{self.random_restarts}:")
            
            perm, b, f = self.run_single_generation_sequence(verbose=verbose)
            
            if f < self.best_overall_f:
                self.best_overall_f = f
                self.best_overall = (perm, b, f)
            
            if verbose:
                print(f"  Restart best: {f:.2f}\n")
        
        print(f"{'='*60}")
        print(f"FINAL BEST FITNESS: {self.best_overall_f:.2f}")
        print(f"{'='*60}\n")
        
        return self.best_overall


# --- Runner: example usage ---
if __name__ == '__main__':
    print("\n" + "="*70)
    print("ENHANCED CIRCLE PACKING WITH EVOLUTIONARY ALGORITHM")
    print("="*70)
    
    # Test problem
    radii = [10, 34, 10, 55, 30, 14, 70, 14]
    weights = [50, 120, 45, 200, 100, 40, 260, 35]
    width = 200.0
    depth = 150.0
    weight_limit = 2000.0

    print("\nProblem Configuration:")
    print(f"  Circles: {len(radii)} items")
    print(f"  Container: {width} × {depth}")
    print(f"  Weight limit: {weight_limit}")
    print(f"  Total weight available: {sum(weights)}")
    
    # Run enhanced EA with multiple restarts
    ea = PackingEA(
        radii, weights, width, depth, weight_limit,
        pop_size=30, generations=150, random_restarts=3
    )
    
    best_perm, best_b, best_f = ea.run(verbose=True)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nBest fitness: {best_f:.2f}")
    
    fit, details = ea.compute_fitness(best_b)
    print(f"\nFitness Breakdown:")
    print(f"  - Overlap penalty: {details['overlap']:.2f}")
    print(f"  - Out-of-bounds penalty: {details['oob']:.2f}")
    print(f"  - Center-of-mass penalty: {details['com_pen']:.2f}")
    print(f"  - Weight penalty: {details['weight_pen']:.2f}")
    print(f"\nCircles placed: {details['num_placed']}/{len(radii)}")
    print(f"Total weight: {details['total_weight']:.0f}/{weight_limit}")
    print(f"Center of mass: ({details['com'][0]:.2f}, {details['com'][1]:.2f})")
    
    print(f"\nLoading order (permutation): {best_perm}")
    
    # Visualize result
    best_b.draw(
        title=f"Optimal Packing (Fitness={best_f:.2f})",
        save_path='best_packing_enhanced.png'
    )
    
    print("\n" + "="*70)
    print("✓ Run complete!")
    print("="*70 + "\n")
