// jreftran_rt.cpp
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <complex>

std::tuple<std::complex<double>, std::complex<double>, double, double, double>
  jreftran_rt(double l, std::vector<double>& d_table, std::vector<std::complex<double>>& n_table, double t0, bool p_polarized)
{
    size_t num = n_table.size();

    std::vector<std::complex<double>> Y; //admittance in terms of impedance of free space and refractive index, assuming non-magnetic media
    Y.reserve(num);
    double Z0(376.730313); //impedance of free space, Ohms
    for (auto& n : n_table) Y.push_back(n/Z0);

    std::vector<std::complex<double>> g; //propagation constant in terms of free space wavelength and refractive index
    g.reserve(num);
    std::complex<double> imunit(0,1);
    for (auto& n : n_table) g.push_back(imunit*2.*M_PI*n/l);

    std::vector<std::complex<double>> ct; // complex theta for each layer
    ct.reserve(num);
    std::complex<double> n1 = n_table.at(0);
    double sint0 = sin(t0);
    for (auto& n : n_table) ct.push_back(sqrt(1.-pow((n1/n*sint0),2)));

    std::vector<std::complex<double>> eta; //tilted admittance
    eta.reserve(num);
    for (size_t j = 0; j < num; j++)
    {
        if (p_polarized)
            eta.push_back(Y[j]/ct[j]); //TM case
        else
            eta.push_back(Y[j]*ct[j]); //TE case
    }

    std::vector<std::complex<double>> delta;
    delta.reserve(num);
    for (size_t j = 0; j < num; j++)
        delta.push_back(imunit * g[j] * d_table[j] * ct[j]);

    std::vector<std::complex<double>> M11, M12, M21, M22;
    M11.reserve(num); M12.reserve(num); M21.reserve(num); M22.reserve(num);
    for (size_t j = 0; j < num; j++)
    {
        auto dj = delta[j];
        M11.push_back(cos(dj));
        M12.push_back(imunit/eta[j]*sin(dj));
        M21.push_back(imunit*eta[j]*sin(dj));
        M22.push_back(cos(dj));
    }

    std::complex<double> Mt11(1,0), Mt12(0,0), Mt21(0,0), Mt22(1,0); //M total
    for (size_t j = 1; j < num-1; j++)
    {
        auto mt11 = Mt11*M11[j] + Mt12*M21[j];
        auto mt12 = Mt11*M12[j] + Mt12*M22[j];
        auto mt21 = Mt21*M11[j] + Mt22*M21[j];
        auto mt22 = Mt21*M12[j] + Mt22*M22[j];
        Mt11 = mt11; Mt12 = mt12;
        Mt21 = mt21; Mt22 = mt22;
    }

    auto eta1 = eta.front();
    auto etaN = eta.back();
    auto r = (eta1 * (Mt11 + Mt12*etaN) - (Mt21 + Mt22*etaN)) / (eta1 * (Mt11 + Mt12*etaN) + (Mt21 + Mt22*etaN));
    auto t = 2.*eta1 / (eta1 * (Mt11 + Mt12*etaN) + (Mt21 + Mt22*etaN));
    double R = pow(abs(r),2);
    double T = real(etaN/eta1) * pow(abs(t),2);
    double A = (4.*real(eta1) * real((Mt11+Mt12*etaN) * conj(Mt21+Mt22*etaN) - etaN)) / pow(abs(eta1 * (Mt11+Mt12*etaN) + (Mt21+Mt22*etaN)), 2);
    return std::make_tuple(r, t, R, T, A);
}

#ifdef _TEST
#include <iostream>
int main(int argc, char* argv[])
{
    std::vector<double> d_table = { NAN, 200.0, NAN };
    std::vector<std::complex<double>> n_table = { {1.,0}, {0.9707,1.8562}, {1.,0} };
    std::complex<double> r, t;
    double R, T, A;
    std::tie(r, t, R, T, A) = jreftran_rt(500., d_table, n_table, 0., false);
    std::cout << "r = " << r.real() << "+" << r.imag() << "i" << std::endl;
    std::cout << "t = " << t.real() << "+" << t.imag() << "i" << std::endl;
    std::cout << "R = " << R << std::endl;
    std::cout << "T = " << T << std::endl;
    std::cout << "A = " << A << std::endl;
//output:
// r = -0.4622 - 0.5066i
// t = -0.0047 + 0.0097i
// R = 0.4702
// T = 1.1593e-04
// A = 0.5296
    return 0;
}
#endif
