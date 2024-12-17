#include <iostream>
#include <sstream>
#include <fstream>
#include <time.h>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>
#include <cassert>
#include <math.h>
#include <omp.h>


struct XSFData
{
    std::vector<double> W;
    std::array<int, 3> n_size;
    std::array<double, 3> origin;
    std::array<std::array<double, 3>, 3> vecs;
};

XSFData xsf_parser(const std::string &filename)
{

    XSFData data;
    std::string line;

    std::ifstream file(filename);


    while (std::getline(file, line))
    {
        if (line.find("BEGIN_DATAGRID_3D_UNKNOWN") != std::string::npos)
        {
            break;
        }
    }

    // Read n_size
    if (!std::getline(file, line)) {
        throw std::runtime_error("Unexpected end of file while reading n_size.");
    }
    std::istringstream iss(line);
    for (int& n : data.n_size) {
        iss >> n;
    }

    if (!std::getline(file, line)) {
        throw std::runtime_error("Unexpected end of file while reading origin.");
    }
    iss.clear();
    iss.str(line);
    for (double& o : data.origin) {
        iss >> o;
    }

    // Read vectors
    for (auto& vec : data.vecs) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("Unexpected end of file while reading vectors.");
        }
        iss.clear();
        iss.str(line);
        for (double& v : vec) {
            iss >> v;
        }
    }

    // Read W data
    double value;
    while (file >> value)
    {
        data.W.push_back(value);
    }
    std::cout << "File " << filename << " was scanned successfully" << std::endl;

    return data;
}



void normalize(std::vector<double> &W)
{
    int n_tot = W.size();
    double norm = 0.0;
    for (int i = 0; i < n_tot; ++i)
    {
        norm += W[i] * W[i];
    }

    for (int i = 0; i < n_tot; ++i)
    {
        W[i] = W[i] / sqrt(norm);
    }
}



template <size_t N>
auto distance(const std::array<double, N> &a, const std::array<double, N> &b)
{
    double d = 0.0;
    for (int i = 0; i < N; ++i)
    {
        d += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(d);
}



std::tuple<std::vector<std::array<double, 3>>, std::vector<double>, std::vector<double>>
size_reduction(const std::vector<double>& W1, const std::vector<double>& W2, 
               const std::array<int, 3>& n_size, 
               const std::array<std::array<double, 3>, 3>& vecs, 
               const std::array<double, 3>& r_center, double r_cut) {
    
    int n_tot = n_size[0] * n_size[1] * n_size[2];

    std::vector<double> W1_new;
    std::vector<double> W2_new;
    std::vector<std::array<double, 3>> r_new;

    double norm_1 = 0.0;
    double norm_2 = 0.0;

    for (int i = 0; i < n_tot; ++i) {
        int c = i / (n_size[0] * n_size[1]);
        int a = (i - (n_size[0] * n_size[1]) * c) % n_size[0];
        int b = (i - (n_size[0] * n_size[1]) * c) / n_size[0];


        std::array<double, 3> r = {
            (vecs[0][0] * a) / n_size[0] + (vecs[1][0] * b) / n_size[1] + (vecs[2][0] * c) / n_size[2],
            (vecs[0][1] * a) / n_size[0] + (vecs[1][1] * b) / n_size[1] + (vecs[2][1] * c) / n_size[2],
            (vecs[0][2] * a) / n_size[0] + (vecs[1][2] * b) / n_size[1] + (vecs[2][2] * c) / n_size[2]
        };



        if (distance(r_center, r) < r_cut)
        {
            W1_new.push_back(W1[i]);
            W2_new.push_back(W1[i]);
            r_new.push_back(r);

            norm_1 += W1[i] * W1[i];
            norm_2 += W2[i] * W2[i];
        }
    }

    int n_tot_new = W1_new.size();
    int size_reduction_percent = static_cast<int>(100 * (n_tot - n_tot_new) / n_tot);

    std::cout << "Size reduction leads to W1 and W2 norms: " 
              << norm_1 << ", " << norm_2 << std::endl;
    std::cout << "Size reduction factor: " << size_reduction_percent << "%" << std::endl;
    std::cout << "WARNING: norms after size reduction should not be far from 1!!!" << std::endl;

    return {r_new, W1_new, W2_new};
}




std::array<double, 3> compute_Coulomb(const int mc_steps, const size_t n_tot, std::vector<double> const &W1, std::vector<double> const &W2, std::vector<std::array<double, 3>> const &r)
{

    double coulomb_U = 0.0;
    double coulomb_V = 0.0;
    double coulomb_J = 0.0;
    srand(time(NULL));

#pragma omp parallel for default(none) shared(mc_steps, n_tot, W1, W2, r) reduction(+ : coulomb_U, coulomb_V, coulomb_J)
    for (auto n = size_t{0}; n < mc_steps; ++n)
    {
        double local_coulomb_U = 0.0;
        double local_coulomb_V = 0.0;
        double local_coulomb_J = 0.0;

        int i = rand() % n_tot;
        int j = rand() % n_tot;
        if (i != j)
        {
            double d_ij = distance(r[i], r[j]);
            local_coulomb_U += (W1[i] * W1[i]) * (W1[j] * W1[j]) / d_ij;
            local_coulomb_V += (W1[i] * W1[i]) * (W2[j] * W2[j]) / d_ij;
            local_coulomb_J += (W1[i] * W2[i]) * (W1[j] * W2[j]) / d_ij;
        }

        coulomb_U += local_coulomb_U;
        coulomb_V += local_coulomb_V;
        coulomb_J += local_coulomb_J;
    }

    coulomb_U *= 14.3948 * (n_tot * n_tot / mc_steps);
    coulomb_V *= 14.3948 * (n_tot * n_tot / mc_steps);
    coulomb_J *= 14.3948 * (n_tot * n_tot / mc_steps);

    return {coulomb_U, coulomb_V, coulomb_J};
}

int main()
{
    // play with this parameter to reach the required accuracy
    const int mc_steps = 1E9;
    time_t td;
    td = time(NULL);

    // set the center r_center for size reduction
    // set the cutoff distance to increase the accuracy of MC sampling
    //  keep in mind that norm_1 and norm_2 should be close to 1 after size reduction!!!
    std::array<double, 3> r_center = {0, 0, 9.5176};
    double r_cut = 25;

    std::cout << "Program Wannier_Hund.x v.2.0 starts on " << ctime(&td);
    std::cout << "=====================================================================" << std::endl;

    std::cout << "mc_steps: " << mc_steps << std::endl;

    auto [W1, n_size, origin, vecs] = xsf_parser("WF1.xsf");
    auto [W2, n_size2, origin2, vecs2] = xsf_parser("WF2.xsf");


    normalize(W1);
    normalize(W2);

    std::cout << "Dimensions are: " << n_size[0] << " " << n_size[1] << " " << n_size[2] << std::endl;
    std::cout << "Origin is: " 
              << std::fixed << std::setprecision(3)
              << origin[0] << " " << origin[1] << " " << origin[2] << std::endl;

    
    std::cout << "Span_vectors are:" << std::endl;
    for (const auto& vec : vecs) {
        std::cout << std::fixed << std::setprecision(3)
                  << vec[0] << " " << vec[1] << " " << vec[2] << std::endl;
    }

    auto [r_new, W1_new, W2_new] = size_reduction(W1, W2, n_size, vecs, r_center, r_cut);

    int n_tot_new = W1_new.size();

    std::array<double, 3> coulomb = compute_Coulomb(mc_steps, n_tot_new, W1_new, W2_new, r_new);

    std::cout << "Coulomb_U: " << coulomb[0] << " eV" << std::endl;
    std::cout << "Coulomb_V: " << coulomb[1] << " eV" << std::endl;
    std::cout << "Coulomb_J: " << coulomb[2] << " eV" << std::endl;
    std::cout << std::endl
              << "=====================================================================" << std::endl;

    td = time(NULL);
    std::cout << "This run was terminated on: " << ctime(&td) << std::endl;
    std::cout << "JOB DONE" << std::endl;
    std::cout << "=====================================================================" << std::endl;
}
