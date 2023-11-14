#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <chrono>

using namespace std;

int main(int argc, char** argv){
    int n = atoi(argv[1]);
    std::vector<std::vector<double>> A(n,std::vector<double>(n, 0));
    std::vector<std::vector<double>> A_new(n,std::vector<double>(n, 0));
    for(int i=0; i < n; ++i){
        for(int j=0; j<n; ++j){
            A[i][j]= sin(i*i+j)*sin(i*i+j)+cos(i-j);
            A_new[i][j]=A[i][j];
            //cout << A[i][j] << " ";
        }
        //cout << endl;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int iter=0; iter < 10; ++iter){
        for(int i=0; i < n; ++i){
            for(int j=0; j<n; ++j){
                if(i==0 || i == n-1 || j==0 || j== n-1){
                    continue;
                }
                std::vector<double> vals = {A[i][j],A[i+1][j],A[i-1][j],A[i][j-1],A[i][j+1]};
                std::sort(vals.begin(),vals.end(),std::greater<double>{});
                A_new[i][j] = vals[2];
            }
        }
        /*
        for(int i=0; i < n; ++i){
            for(int j=0; j<n; ++j){
                cout << A_new[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
        */
        A.swap(A_new);
    }
    double sum = 0;
    for(int i = 0; i < n; ++i){
        double temp_sum = 0;
        for(int j = 0; j < n; ++j){
            temp_sum += A[i][j];
        }
        sum += temp_sum;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "f() took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10 - 1) << sum << " " << A[n/3][n/3] << " " << A[19][37] << std::endl;
}