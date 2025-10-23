#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
// Host entry: read CSV (i,j,w) -> CSR, launch kernel, return mean gain.
int phase14_gain_from_csr(const char* csv_path, int N, double* out_alpha_eff,
                          double gain_base, double gain_span);
}

struct Edge { int i,j; float w; };

static void die(const char* m){ std::fprintf(stderr,"%s\n",m); std::exit(1); }

// ---- CSV (i,j,w) -> CSR (undirected) ----
static void load_csv_undirected(const char* path, int N,
    std::vector<int>& row_ptr, std::vector<int>& col_idx, std::vector<float>& w)
{
    FILE* f = std::fopen(path, "r");
    if(!f) die("[phase14_gain] cannot open J csv");
    std::vector<Edge> edges; edges.reserve(1<<20);
    char buf[256];
    while (std::fgets(buf, sizeof(buf), f)) {
        if(buf[0]=='#' || std::strlen(buf)<3) continue;
        int i,j; float ww;
        if (std::sscanf(buf, "%d,%d,%f", &i,&j,&ww)==3) {
            if(i>=0 && i<N && j>=0 && j<N && ww!=0.0f) {
                edges.push_back({i,j,ww});
                edges.push_back({j,i,ww}); // make symmetric
            }
        }
    }
    std::fclose(f);

    row_ptr.assign(N+1, 0);
    for (auto &e: edges) row_ptr[e.i+1]++;

    for (int i=1;i<=N;i++) row_ptr[i]+=row_ptr[i-1];
    int M = row_ptr[N];
    col_idx.assign(M,0); w.assign(M,0);

    std::vector<int> cursor(row_ptr.begin(), row_ptr.end());
    for (auto &e: edges) {
        int pos = cursor[e.i]++;
        col_idx[pos] = e.j; w[pos] = e.w;
    }
}

// ---- CUDA kernel: per-node alignment -> gain in [base, base+span] ----
__global__ void k_compute_gains(int N, const int* __restrict__ row_ptr,
                                const int* __restrict__ col_idx,
                                const float* __restrict__ w,
                                const float* __restrict__ state,
                                float base, float span, float* __restrict__ gains)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=N) return;
    int start = row_ptr[i], end = row_ptr[i+1];
    float zi = state[i];
    float num = 0.f, den = 0.f;
    for (int p=start;p<end;p++) {
        int j = col_idx[p];
        float wij = w[p];
        num += wij * zi * state[j]; // alignment on ZZ
        den += fabsf(wij);
    }
    float align = (den>0.f) ? 0.5f*(num/den + 1.f) : 0.5f; // map [-1,1]->[0,1]
    float alpha = base + span * align;                     // [base, base+span]
    gains[i] = alpha;
}

extern "C"
int phase14_gain_from_csr(const char* csv_path, int N, double* out_alpha_eff,
                          double gain_base, double gain_span)
{
    if(!csv_path || !out_alpha_eff) return -1;

    // Build CSR on host
    std::vector<int> h_row_ptr, h_col_idx; std::vector<float> h_w;
    load_csv_undirected(csv_path, N, h_row_ptr, h_col_idx, h_w);
    int M = (int)h_col_idx.size();

    // Default state: +1 for all nodes (can be replaced later)
    std::vector<float> h_state(N, +1.0f);

    // Device alloc
    int *d_row_ptr=nullptr,*d_col_idx=nullptr; float *d_w=nullptr,*d_state=nullptr,*d_gains=nullptr;
    cudaMalloc(&d_row_ptr,(N+1)*sizeof(int));
    cudaMalloc(&d_col_idx,M*sizeof(int));
    cudaMalloc(&d_w,M*sizeof(float));
    cudaMalloc(&d_state,N*sizeof(float));
    cudaMalloc(&d_gains,N*sizeof(float));
    cudaMemcpy(d_row_ptr,h_row_ptr.data(),(N+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx,h_col_idx.data(),M*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_w,h_w.data(),M*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_state,h_state.data(),N*sizeof(float),cudaMemcpyHostToDevice);

    // Launch
    dim3 blk(256), grd((N+255)/256);
    k_compute_gains<<<grd,blk>>>(N,d_row_ptr,d_col_idx,d_w,d_state,
                                 (float)gain_base,(float)gain_span,d_gains);
    cudaDeviceSynchronize();

    // Collect and average
    std::vector<float> h_gains(N);
    cudaMemcpy(h_gains.data(), d_gains, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum=0.0; for (int i=0;i<N;i++) sum += (double)h_gains[i];
    *out_alpha_eff = sum / (double)N;

    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_w);
    cudaFree(d_state);   cudaFree(d_gains);
    return 0;
}