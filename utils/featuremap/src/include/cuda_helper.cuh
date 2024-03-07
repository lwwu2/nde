template <typename T>
inline __device__ __host__ T div_round_up(T a, T b) {
    return (a+b-1)/b;
}