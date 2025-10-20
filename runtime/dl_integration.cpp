#include "dl_integration.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <string>

#ifdef USE_LIBTORCH
#include <torch/script.h>
#include <torch/torch.h>
#endif

namespace {
std::mutex g_mutex;
std::string g_last_error;
std::atomic<bool> g_loaded{false};
int g_prefer_gpu = 1;

#ifdef USE_LIBTORCH
std::unique_ptr<torch::jit::Module> g_module;
torch::Device g_device(torch::kCPU);
#endif

void set_error(const std::string& err) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_last_error = err;
}
}  // namespace

int dl_model_supported(void) {
#ifdef USE_LIBTORCH
    return 1;
#else
    return 0;
#endif
}

int dl_model_is_loaded(void) {
    return g_loaded.load();
}

const char* dl_model_last_error(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_last_error.c_str();
}

int dl_model_load(const char* model_path, int prefer_gpu) {
    if (!model_path || *model_path == '\0') {
        set_error("Model path is empty");
        return -1;
    }
    g_prefer_gpu = prefer_gpu ? 1 : 0;

#ifdef USE_LIBTORCH
    try {
        torch::jit::script::Module module = torch::jit::load(model_path);
        torch::Device device(torch::kCPU);
        if (g_prefer_gpu && torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
            module.to(device);
        }
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_module = std::make_unique<torch::jit::script::Module>(std::move(module));
            g_device = device;
            g_last_error.clear();
        }
        g_loaded.store(true);
        return 0;
    } catch (const c10::Error& e) {
        set_error(std::string("Failed to load model: ") + e.what_without_backtrace());
    } catch (const std::exception& e) {
        set_error(std::string("Failed to load model: ") + e.what());
    }
    g_loaded.store(false);
    return -1;
#else
    (void)prefer_gpu;
    set_error("Binary not built with LibTorch support (rebuild with USE_LIBTORCH=1)");
    g_loaded.store(false);
    return -1;
#endif
}

int dl_model_infer(const float* input, int input_len, float* output, int output_len) {
    if (!g_loaded.load()) {
        set_error("No model loaded");
        return -1;
    }
    if (!input || input_len <= 0 || !output || output_len <= 0) {
        set_error("Invalid buffers");
        return -2;
    }

#ifdef USE_LIBTORCH
    try {
        std::unique_lock<std::mutex> lock(g_mutex);
        auto* module = g_module.get();
        if (!module) {
            set_error("Model pointer null");
            g_loaded.store(false);
            return -3;
        }

        torch::Tensor input_tensor = torch::from_blob(
            const_cast<float*>(input),
            {1, input_len},
            torch::TensorOptions().dtype(torch::kFloat32));
        input_tensor = input_tensor.clone().to(g_device);

        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(input_tensor);

        torch::NoGradGuard guard;
        torch::jit::IValue result = module->forward(inputs);
        lock.unlock();

        torch::Tensor output_tensor;
        if (result.isTensor()) {
            output_tensor = result.toTensor();
        } else if (result.isTuple()) {
            auto tuple = result.toTuple();
            if (tuple->elements().empty() || !tuple->elements()[0].isTensor()) {
                set_error("Model output tuple does not contain a tensor");
                return -4;
            }
            output_tensor = tuple->elements()[0].toTensor();
        } else {
            set_error("Model output not a tensor");
            return -4;
        }

        output_tensor = output_tensor.to(torch::kCPU).contiguous();
        int64_t out_elems = output_tensor.numel();
        if (out_elems > output_len) {
            set_error("Output buffer too small for model result");
            return -5;
        }
        std::memcpy(output, output_tensor.data_ptr<float>(), sizeof(float) * out_elems);
        return static_cast<int>(out_elems);
    } catch (const c10::Error& e) {
        set_error(std::string("Inference error: ") + e.what_without_backtrace());
    } catch (const std::exception& e) {
        set_error(std::string("Inference error: ") + e.what());
    }
    return -6;
#else
    set_error("Binary not built with LibTorch support (rebuild with USE_LIBTORCH=1)");
    return -1;
#endif
}

void dl_model_unload(void) {
#ifdef USE_LIBTORCH
    std::lock_guard<std::mutex> lock(g_mutex);
    g_module.reset();
#endif
    g_loaded.store(false);
}
