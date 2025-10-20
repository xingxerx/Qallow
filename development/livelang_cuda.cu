// Compile: nvcc livelang_cuda.cu -o livelang_cuda
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define MAX_LEN 256
static std::vector<std::string> prog;
static int indent = 1;

static void add_line(const std::string& s){ prog.push_back(s); }

static void render_c() {
    FILE* f = fopen("output.c","w");
    if(!f) return;
    fprintf(f, "#include <stdio.h>\n\nint main(){\n");
    for (auto& s: prog){ for(int j=0;j<indent;j++) fprintf(f,"    "); fprintf(f,"%s\n", s.c_str()); }
    fprintf(f,"    return 0;\n}\n"); fclose(f);
}

static void compile_and_run() {
    system("gcc output.c -o output 2> /dev/null");
    printf("\n--- Running output ---\n");
    system("./output");
    printf("\n----------------------\n");
}

/* ================= CUDA helpers ================= */
#include <cuda_runtime.h>
#define CUCHK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return; } }while(0)

__global__ void k_add(const int* __restrict__ A, const int* __restrict__ B, int* __restrict__ C, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) C[i] = A[i] + B[i];
}

__global__ void k_sum_atomic(unsigned long long* acc, int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) atomicAdd(acc, (unsigned long long)i);
}

static void gpu_add(int N){
    int *dA=0,*dB=0,*dC=0;
    CUCHK(cudaMalloc(&dA, N*sizeof(int)));
    CUCHK(cudaMalloc(&dB, N*sizeof(int)));
    CUCHK(cudaMalloc(&dC, N*sizeof(int)));

    std::vector<int> h(N); for(int i=0;i<N;i++) h[i]=i;
    CUCHK(cudaMemcpy(dA, h.data(), N*sizeof(int), cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(dB, h.data(), N*sizeof(int), cudaMemcpyHostToDevice));

    int bs=256, gs=(N+bs-1)/bs;
    k_add<<<gs,bs>>>(dA,dB,dC,N);
    CUCHK(cudaPeekAtLastError());
    CUCHK(cudaDeviceSynchronize());

    int last=0; CUCHK(cudaMemcpy(&last, dC+N-1, sizeof(int), cudaMemcpyDeviceToHost));
    printf("[GPU] add: C[0]=%d, C[%d]=%d\n", 0, N-1, last);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

static void gpu_sum(int N){
    unsigned long long *dAcc=0;
    CUCHK(cudaMalloc(&dAcc, sizeof(unsigned long long)));
    CUCHK(cudaMemset(dAcc, 0, sizeof(unsigned long long)));
    int bs=256, gs=(N+bs-1)/bs;
    k_sum_atomic<<<gs,bs>>>(dAcc,N);
    CUCHK(cudaPeekAtLastError());
    CUCHK(cudaDeviceSynchronize());
    unsigned long long res=0; CUCHK(cudaMemcpy(&res,dAcc,sizeof(res),cudaMemcpyDeviceToHost));
    unsigned long long ref = (unsigned long long)N*(unsigned long long)(N-1)/2;
    printf("[GPU] sum 0..%d-1 = %llu (ref %llu)\n", N, res, ref);
    cudaFree(dAcc);
}
/* ================================================= */

static void handle_line(const char* in) {
    if (strncmp(in,"gpu.add ",8)==0){
        int N=0; if(sscanf(in+8,"%d",&N)==1 && N>0){ gpu_add(N); return; }
        printf("[GPU] usage: gpu.add N\n"); return;
    }
    if (strncmp(in,"gpu.sum ",8)==0){
        int N=0; if(sscanf(in+8,"%d",&N)==1 && N>0){ gpu_sum(N); return; }
        printf("[GPU] usage: gpu.sum N\n"); return;
    }

    if (strncmp(in,"print ",6)==0){
        const char* msg = in+6;
        char esc[MAX_LEN];
        snprintf(esc,sizeof(esc), "printf(\"%s\\n\");", msg);
        add_line(esc);
    } else if (strncmp(in,"let ",4)==0){
        char var[MAX_LEN]; int val;
        if (sscanf(in+4,"%s = %d", var,&val)==2) {
            char out[MAX_LEN]; snprintf(out,sizeof(out),"int %s = %d;",var,val); add_line(out);
        }
    } else if (strncmp(in,"add ",4)==0){
        char var[MAX_LEN]; int val;
        if (sscanf(in+4,"%s %d", var,&val)==2) {
            char out[MAX_LEN]; snprintf(out,sizeof(out),"%s += %d;",var,val); add_line(out);
        }
    } else if (strncmp(in,"for ",4)==0){
        char var[MAX_LEN]; int end;
        if (sscanf(in+4,"%s in 0..<%d", var,&end)==2){
            char out[MAX_LEN];
            snprintf(out,sizeof(out),"for (int %s=0; %s<%d; ++%s) {",var,var,end,var);
            add_line(out); indent++;
        }
    } else if (strncmp(in,"dedent",6)==0){
        indent = indent>1? indent-1:1; add_line("}");
    } else {
        char out[MAX_LEN]; snprintf(out,sizeof(out),"/* Unknown: %s */", in); add_line(out);
    }

    render_c();
    compile_and_run();
}

int main(){
    printf("LiveLang CUDA\n");
    printf("CPU→C: print | let | add | for i in 0..<N: | dedent (auto-compile+run)\n");
    printf("GPU ops: gpu.add N | gpu.sum N (immediate on GPU)\n\n");

    char buf[MAX_LEN];
    while (fgets(buf,sizeof(buf),stdin)){
        buf[strcspn(buf,"\n")] = 0;
        if(buf[0]==0) continue;
        handle_line(buf);
    }
    return 0;
}
