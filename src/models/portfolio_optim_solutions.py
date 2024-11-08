


class POETSolution:

    def __init__(self, daily_returns_vec, k_estimate_vec, C_min_estimate_vec, M_estimate_vec, C_star_vec, t_elap_vec):
        self.daily_returns_vec = daily_returns_vec
        self.k_estimate_vec = k_estimate_vec
        self.C_min_estimate_vec = C_min_estimate_vec
        self.M_estimate_vec = M_estimate_vec
        self.C_star_vec = C_star_vec
        self.t_elap_vec = t_elap_vec


class BasicSolverSolution:

    def __init__(self,daily_returns_vec):
        self.daily_returns_vec = daily_returns_vec