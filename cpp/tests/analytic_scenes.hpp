// Scene generators shared by the CPU and CUDA analytic tests.
//
// Shared deliberately: the cross-arm check is only meaningful if both arms see
// bit-identical input. Two copies of "the same" generator would drift.
#pragma once

#include <cmath>
#include <vector>

namespace scenes {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/// Points on a sphere, as seen from a camera at the centre looking outward.
/// From inside, every surface point is visible from one viewpoint, so the sign
/// convention is exercised without needing multiple views.
inline std::vector<float> sphere(float radius, int n_theta, int n_phi) {
    std::vector<float> p;
    p.reserve(static_cast<std::size_t>(n_theta) * n_phi * 3);
    for (int i = 0; i < n_theta; ++i) {
        const float theta = static_cast<float>(M_PI) * (i + 0.5f) / n_theta;
        for (int j = 0; j < n_phi; ++j) {
            const float phi = 2.0f * static_cast<float>(M_PI) * j / n_phi;
            p.push_back(radius * std::sin(theta) * std::cos(phi));
            p.push_back(radius * std::sin(theta) * std::sin(phi));
            p.push_back(radius * std::cos(theta));
        }
    }
    return p;
}

/// A z = z0 plane patch, seen from a camera below it.
inline std::vector<float> plane(float z0, float half_extent, int n) {
    std::vector<float> p;
    p.reserve(static_cast<std::size_t>(n) * n * 3);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            p.push_back(-half_extent + 2.0f * half_extent * i / (n - 1));
            p.push_back(-half_extent + 2.0f * half_extent * j / (n - 1));
            p.push_back(z0);
        }
    }
    return p;
}

}  // namespace scenes
