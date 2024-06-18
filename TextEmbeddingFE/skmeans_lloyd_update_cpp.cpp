#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

extern "C" {
    void skmeans_lloyd_update_cpp(double* X, double* centroids, double* similarities, int n_obs, int n_clusters, int n_features, 
                              int* new_labels, double* new_centroids, double* frobenius_norm) {
        std::fill(new_labels, new_labels + n_obs, 0);
        std::fill(new_centroids, new_centroids + n_clusters * n_features, 0.0);

        for (int n = 0; n < n_obs; ++n) {
            int best_cluster = 0;
            double best_similarity = similarities[n * n_clusters];

            for (int k = 1; k < n_clusters; ++k) {
                if (similarities[n * n_clusters + k] > best_similarity) {
                    best_similarity = similarities[n * n_clusters + k];
                    best_cluster = k;
                }
            }

            new_labels[n] = best_cluster;
            for (int f = 0; f < n_features; ++f) {
                new_centroids[best_cluster * n_features + f] += X[n * n_features + f];
            }
        }

        std::vector<double> norms(n_clusters, 0.0);
        for (int k = 0; k < n_clusters; ++k) {
            double norm = 0.0;
            for (int f = 0; f < n_features; ++f) {
                norm += new_centroids[k * n_features + f] * new_centroids[k * n_features + f];
            }
            norm = std::sqrt(norm);
            //if (norm == 0) {
            //    throw std::runtime_error("One or more clusters are empty");
            //}
            norms[k] = norm;
        }

        for (int k = 0; k < n_clusters; ++k) {
            for (int f = 0; f < n_features; ++f) {
                new_centroids[k * n_features + f] /= norms[k];
            }
        }

        std::fill(similarities, similarities + n_obs * n_clusters, 0.0);
        for (int n = 0; n < n_obs; ++n) {
            for (int k = 0; k < n_clusters; ++k) {
                double dot_product = 0.0;
                for (int f = 0; f < n_features; ++f) {
                    dot_product += X[n * n_features + f] * new_centroids[k * n_features + f];
                }
                similarities[n * n_clusters + k] = dot_product;
            }
        }

        double sum_of_squares = 0.0;
        for (int k = 0; k < n_clusters; ++k) {
            for (int f = 0; f < n_features; ++f) {
                double diff = new_centroids[k * n_features + f] - centroids[k * n_features + f];
                sum_of_squares += diff * diff;
            }
        }

        *frobenius_norm = std::sqrt(sum_of_squares);
    }
}
