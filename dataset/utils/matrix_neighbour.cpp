#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Funkcja generująca krawędzie
std::tuple<std::vector<int>, std::vector<std::tuple<int, int, int>>, std::vector<std::pair<int, int>>>
generate_edges(const std::vector<std::vector<int>>& events, int radius, int width, int height) {
    int num_nodes = events.size();

    std::vector<std::pair<int, int>> edges;
    edges.reserve(num_nodes);
    std::vector<std::tuple<int, int, int>> positions;
    positions.reserve(num_nodes);
    std::vector<int> features;
    features.reserve(num_nodes);

    std::vector<std::vector<int>> neigh_matrix(width, std::vector<int>(height, -1));
    std::vector<std::vector<int>> idx_matrix(width, std::vector<int>(height, -1));
    int idx = 0;

    int radius_sq = radius * radius;

    for (int i = 0; i < num_nodes; ++i) {
        int x = events[i][0];
        int y = events[i][1];
        int t = events[i][2];
        int p = events[i][3];

        // Sprawdzenie duplikatów
        if (t == neigh_matrix[x][y]) continue;

        // Dodanie pętli własnej
        edges.emplace_back(idx, idx);
        features.push_back(p);
        positions.emplace_back(x, y, t);

        int x_start = std::max(0, x - radius);
        int x_end = std::min(width - 1, x + radius);
        int y_start = std::max(0, y - radius);
        int y_end = std::min(height - 1, y + radius);

        // Sprawdzenie sąsiadów w promieniu
        for (int j = x_start; j <= x_end; ++j) {
            for (int k = y_start; k <= y_end; ++k) {
                if (neigh_matrix[j][k] == -1) continue;

                int dx = x - j;
                int dy = y - k;
                int dt = t - neigh_matrix[j][k];
                int dist_sq = dx*dx + dy*dy + dt*dt;
                if (dist_sq <= radius_sq) {
                    edges.emplace_back(idx, idx_matrix[j][k]);
                }
            }
        }

        neigh_matrix[x][y] = t;
        idx_matrix[x][y] = idx;
        idx++;
    }

    return std::make_tuple(features, positions, edges);
}

// Kod wiążący
namespace py = pybind11;

PYBIND11_MODULE(matrix_neighbour, m) {
    m.def("generate_edges", &generate_edges, "Generowanie krawędzi",
          py::arg("events"), py::arg("radius"), py::arg("width"), py::arg("height"));
}