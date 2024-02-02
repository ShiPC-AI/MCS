#include "base/io.h"
void printInfo(const int& idx, const int& step, std::string info) {
    if (idx % step == 0) {
        std::cout << idx << "th " << info << "\n";
    }
}



