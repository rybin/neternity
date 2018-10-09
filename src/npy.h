#include <cnpy.h>

template <typename T> class Npy
{
public:
    std::vector<size_t> shape;
    size_t word_size;
    T* data;

    Npy(cnpy::NpyArray NpyArr) {
        shape = NpyArr.shape;
        word_size = NpyArr.word_size;
        data = NpyArr.data<T>();
    }

    T get(int x, int y, int z) {
        return data[x * shape[1] * shape[1] + y * shape[1] + z];
    }

    T get(int x, int y) {
        return data[x * shape[1]  + y];
    }

    T get(int x) {
        return data[x];
    }
};
