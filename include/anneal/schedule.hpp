// schedule.hpp
//
// Temperature schedules for simulated annealing. A schedule maps a sweep
// index to a temperature; the annealer calls temperature(sweep) once per
// sweep and uses it in the Metropolis acceptance rule. Keeping the
// schedule as a small callable object (rather than a hardcoded formula
// inside the annealer) makes it easy to swap in different cooling
// strategies without touching the solver's inner loop.
#pragma once

#include <cmath>
#include <cstddef>

namespace anneal {

// Geometric cooling: T_k = T0 * alpha^k. The classic simulated-annealing
// schedule (Kirkpatrick et al. 1983). alpha should be a bit below 1
// (e.g. 0.99) so temperature decays slowly enough to explore before
// freezing.
class GeometricSchedule {
public:
    GeometricSchedule(double t0, double alpha) : t0_(t0), alpha_(alpha) {}

    double temperature(std::size_t sweep) const {
        return t0_ * std::pow(alpha_, static_cast<double>(sweep));
    }

private:
    double t0_;
    double alpha_;
};

// Linear cooling: temperature steps evenly from t0 down to t_end over
// num_steps sweeps, then holds at t_end.
class LinearSchedule {
public:
    LinearSchedule(double t0, double t_end, std::size_t num_steps)
        : t0_(t0), t_end_(t_end), num_steps_(num_steps) {}

    double temperature(std::size_t sweep) const {
        if (num_steps_ == 0 || sweep >= num_steps_) return t_end_;
        double frac = static_cast<double>(sweep) / static_cast<double>(num_steps_);
        return t0_ + (t_end_ - t0_) * frac;
    }

private:
    double t0_;
    double t_end_;
    std::size_t num_steps_;
};

}  // namespace anneal
