#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <array>
#include <math.h>
#include <omp.h>

void normalize(std::vector<double> &W, int n_tot)
{
    auto norm = 0.0;
    for (auto i = size_t{0}; i < n_tot; ++i)
    {
        norm += W[i] * W[i];
    }

    for (auto i = size_t{0}; i < n_tot; ++i)
    {
        W[i] = W[i] / sqrt(norm);
    }
}

template <size_t N>
auto distance(const std::array<double, N> &a, const std::array<double, N> &b)
{
    auto d = 0.0;
    for (auto i = size_t{0}; i < N; ++i)
    {
        d += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(d);
}

std::array<double, 3> compute_Coulomb(const int mc_steps, const size_t n_tot, std::vector<double> const &W1, std::vector<double> const &W2, std::vector<std::array<double, 3> > const &r)
{
    
    double coulomb_U = 0.0;
    double coulomb_V = 0.0;
    double coulomb_J = 0.0;
    srand(time(NULL));

    // #pragma omp parallel for default(none) shared(mc_steps, n_tot, W1, W2, r) reduction(+:coulomb_U, coulomb_V, coulomb_J)
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
    //play with this parameter to reach the required accuracy
    const int mc_steps = 1E19;
    int n_tot, n_tot_new, a, b, c;
    int n_size[3];
    double origin[3];
    double vecs[3][3];
    double norm_1, norm_2;
    std::array<double, 3> coulomb;

    std::string point = "BEGIN_DATAGRID_3D_UNKNOWN";  // xsf data file
    std::string line;

    time_t td;
    td = time(NULL);

    //set the center r_center for size reduction
    //set the cutoff distance to increase the accuracy of MC sampling
    // keep in mind that norm_1 and norm_2 should be close to 1 after size reduction!!!
    std::array<double, 3> r_center = {0, 0, 9.5176};
    double r_cut = 25;
    

    std::cout << "Program Wannier_Hund.x v.2.0 starts on " << ctime(&td);
    std::cout << "=====================================================================" << std::endl;

    std::cout << "mc_steps: " << mc_steps << std::endl;

    std::ifstream main;
    main.open("W1.xsf");
    if (!main)
    {
        std::cout << "ERROR!Cannot open file <W1.xsf>!" << std::endl;
        return 0;
    }

    while (getline(main, line) && line.compare(point) != 0) {
    }// go to data block


    main >> n_size[0] >> n_size[1] >> n_size[2];
    std::cout << "Dimensions are: " << n_size[0] << " " << n_size[1] << " " << n_size[2] << std::endl;

    n_tot = n_size[0] * n_size[1] * n_size[2];

    main >> origin[0] >> origin[1] >> origin[2];
    std::cout <<  "Origin is: " << origin[0] << " " << origin[1] << " " << origin[2] << std::endl;

    std::vector<double> W1(n_tot);
    std::vector<double> W1_new;

    std::vector<double> W2(n_tot);
    std::vector<double> W2_new;

    std::vector<std::array<double, 3> > r(n_tot);
    std::vector<std::array<double, 3> > r_new;

    for (auto i = size_t{0}; i < 3; ++i)
    {
        main >> vecs[i][0] >> vecs[i][1] >> vecs[i][2];
    }

    std::cout << "Span_vectors are: " << std::endl;
    for (auto i = size_t{0}; i < 3; ++i)
    {
        std::cout << vecs[i][0] << " " << vecs[i][1] << " " << vecs[i][2] << std::endl;
    }

    for (auto i = size_t{0}; i < n_tot; ++i)
    {
        main >> W1[i];
    }

    std::cout << "File <W1.xsf> was  scanned  successfully" << std::endl;
    main.close();




    main.open("W2.xsf");
    if (!main)
    {
        std::cout << "ERROR!Cannot open file <W2.xsf>!" << std::endl;
        return 0;
    }

    while (getline(main, line) && line.compare(point) != 0) {
    }// go to data block


    main >> n_size[0] >> n_size[1] >> n_size[2];
    main >> origin[0] >> origin[1] >> origin[2];

    for (auto i = size_t{0}; i < 3; ++i)
    {
        main >> vecs[i][0] >> vecs[i][1] >> vecs[i][2];
    }


    for (auto i = size_t{0}; i < n_tot; ++i)
    {
        main >> W2[i];
    }

    std::cout << "File <W2.xsf> was  scanned  successfully" << std::endl;
    main.close();



    normalize(W1, n_tot);
    normalize(W2, n_tot);


    norm_1 = 0.0;
    norm_2 = 0.0;
    for (auto i = size_t{0}; i < n_tot; ++i)
    {
        c = i / (n_size[0] * n_size[1]);
        a = (i - (n_size[0] * n_size[1]) * c) % (n_size[0]);
        b = (i - (n_size[0] * n_size[1]) * c) / (n_size[0]);

        for (auto j = size_t{0}; j < 3; ++j)
        {
            r[i][j] = (vecs[0][j] * a) / n_size[0] + (vecs[1][j] * b) / n_size[1] + (vecs[2][j] * c) / n_size[2];
        }

        if (distance(r_center, r[i]) < r_cut)
        {
            W1_new.push_back(W1[i]);
            W2_new.push_back(W1[i]);
            r_new.push_back(r[i]);

            norm_1 += W1[i] * W1[i];
            norm_2 += W2[i] * W2[i];
        }
    }

    n_tot_new = W1_new.size();
    
    std::cout << "Size reduction  leads to W1 and W2 norms: " << norm_1 <<" " << norm_2 << std::endl;
    std::cout << "Size reduction factor: " << 100 * (n_tot - n_tot_new)/n_tot << "%" << std::endl;
    std::cout << "WARNING: norms after size reduction should not be far from  1!!!" << std::endl;
    
    coulomb = compute_Coulomb(mc_steps, n_tot_new, W1_new, W2_new, r_new); 

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
