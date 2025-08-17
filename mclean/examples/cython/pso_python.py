import numpy as np

class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(low, high) for low, high in bounds])
        self.velocity = np.array([np.random.uniform(-abs(high - low), abs(high - low)) for low, high in bounds])
        self.best_position = self.position.copy()
        self.best_score = float('inf')

    def update_velocity(self, global_best_position, w, c1, c2):
        r1, r2 = np.random.random(2)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        # Ensure particle stays within bounds
        for i in range(len(bounds)):
            self.position[i] = np.clip(self.position[i], bounds[i][0], bounds[i][1])

def pso(func, bounds, num_particles=30, max_iter=100, w=0.5, c1=1.0, c2=2.0, tol=1e-6, stagnation_iter=20, *args):
    particles = [Particle(bounds) for _ in range(num_particles)]
    global_best_position = particles[0].position.copy()
    global_best_score = float('inf')
    no_improvement_count = 0

    for iter in range(max_iter):
        for particle in particles:
            score = func(particle.position[0], *args)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
            if score < global_best_score:
                global_best_score = score
                global_best_position = particle.position.copy()
                no_improvement_count = 0  # Reset the stagnation counter
            else:
                no_improvement_count += 1

        if no_improvement_count >= stagnation_iter:
            #print(f"No improvement for {stagnation_iter} iterations, stopping early.")
            break

        if global_best_score < tol:
            #print(f"Global best score {global_best_score} reached tolerance {tol}, stopping early.")
            break

        for particle in particles:
            particle.update_velocity(global_best_position, w, c1, c2)
            particle.update_position(bounds)

        #print(f"Iteration {iter+1}/{max_iter}, Global Best Score: {global_best_score}")

    return global_best_position, global_best_score
