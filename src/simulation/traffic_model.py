import numpy as np


class Approach:
    def __init__(self, name, arrival_rate, saturation_flow, phase_index):
        self.name = name
        self.arrival_rate = arrival_rate
        self.saturation_flow = saturation_flow
        self.phase_index = phase_index


class Intersection:

    def __init__(self, name, approaches,
                 yellow_time=4.0,
                 min_green=10.0,
                 max_green=60.0):

        self.name = name
        self.approaches = approaches
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green

        self.n_phases = 0
        for a in approaches:
            if a.phase_index + 1 > self.n_phases:
                self.n_phases = a.phase_index + 1

        self.lost_time = yellow_time * self.n_phases


    def simulate_delay(self, green_times, sim_time=300, dt=5.0):

        green_times = np.clip(green_times, self.min_green, self.max_green)

        cycle = sum(green_times) + self.lost_time

        # Phase start times within cycle
        phase_starts = [0.0]
        for g in green_times[:-1]:
            phase_starts.append(phase_starts[-1] + g)
        # Last phase ends at sum(green_times), then lost_time

        queues = [0.0 for _ in self.approaches]
        total_delay = 0.0
        total_vehicles = 0.0

        t = 0.0
        while t < sim_time:
            cycle_time = t % cycle

            # Determine current green phase
            green_phase = None
            cumulative = 0.0
            for p in range(self.n_phases):
                cumulative += green_times[p]
                if cycle_time < cumulative:
                    green_phase = p
                    break

            for i, a in enumerate(self.approaches):
                # Arrivals
                arrivals = a.arrival_rate * dt
                queues[i] += arrivals
                total_vehicles += arrivals

                # Discharge if green
                if green_phase == a.phase_index:
                    discharge = min(queues[i], a.saturation_flow * dt)
                    queues[i] -= discharge

                # Delay: all vehicles in queue are waiting
                total_delay += queues[i] * dt

            t += dt

        return total_delay / (total_vehicles + 1e-9)


    def compute_delay(self, green_times):

        # Fallback to analytical for speed, but we'll use simulate_delay
        return self.simulate_delay(green_times)


    def n_variables(self):
        return self.n_phases

    def default_green_times(self):

        available = 90.0 - self.lost_time
        per_phase = available / self.n_phases

        # clamp manually
        if per_phase < self.min_green:
            per_phase = self.min_green
        if per_phase > self.max_green:
            per_phase = self.max_green

        result = []
        for _ in range(self.n_phases):
            result.append(per_phase)

        return np.array(result)


class TrafficNetwork:

    def __init__(self, intersections):

        self.intersections = intersections

        self.offsets = []
        offset = 0

        for inter in intersections:
            self.offsets.append(offset)
            offset += inter.n_variables()

        self.n_variables = offset


    def evaluate(self, solution):

        total = 0.0

        for i in range(len(self.intersections)):

            inter = self.intersections[i]
            start = self.offsets[i]
            end = start + inter.n_variables()

            green_times = solution[start:end]

            total += inter.simulate_delay(green_times)

        return total / len(self.intersections)


    def repair(self, solution):

        lb, ub = self.bounds()
        return np.clip(solution, lb, ub)

    def bounds(self):

        lb = []
        ub = []

        for inter in self.intersections:
            for _ in range(inter.n_variables()):
                lb.append(inter.min_green)
                ub.append(inter.max_green)

        return np.array(lb), np.array(ub)

    def baseline_solution(self):

        parts = []

        for inter in self.intersections:
            parts.append(inter.default_green_times())

        return np.concatenate(parts)

    def baseline_fitness(self):
        return self.evaluate(self.baseline_solution())

    def n_intersections(self):
        return len(self.intersections)


def build_single_intersection():

    approaches = [
        Approach("North", 0.40, 1.80, 0),
        Approach("South", 0.35, 1.80, 0),
        Approach("East",  0.50, 1.80, 1),
        Approach("West",  0.45, 1.80, 1),
    ]

    return TrafficNetwork([
        Intersection("INT-1", approaches)
    ])


def build_multi_intersection(n=4):

    rng = np.random.default_rng(42)
    intersections = []

    for k in range(n):

        base = 0.30 + 0.20 * rng.random()

        approaches = [
            Approach("N"+str(k), base + 0.05 * rng.random(), 1.8, 0),
            Approach("S"+str(k), base + 0.05 * rng.random(), 1.8, 0),
            Approach("E"+str(k), base + 0.10 * rng.random(), 1.8, 1),
            Approach("W"+str(k), base + 0.10 * rng.random(), 1.8, 1),
        ]

        intersections.append(Intersection("INT-"+str(k+1), approaches))

    return TrafficNetwork(intersections)
