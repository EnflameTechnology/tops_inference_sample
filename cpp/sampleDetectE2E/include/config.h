#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <memory>
#include <string>
#include <vector>

#include "include/module.h"

namespace e2e_sample {

struct TopsRuntimeResource {
    void* stream = nullptr;
    void* start_event = nullptr;
    void* end_event = nullptr;
};

struct FFmpegGCUConfig {
    std::string name;
    std::string addr;
    int fer; //!frame_extraction_rate
    int card_id;
    int dev_id;
    bool reopen;
    FFmpegGCUConfig() {}
    FFmpegGCUConfig(std::string _name, std::string _addr, int _fer, int _card_id, int _dev_id, int _reopen) : name(_name), addr(_addr), fer(_fer), card_id(_card_id), dev_id(_dev_id), reopen(_reopen){}
};

struct MultiThreadsModuleConfig {
    int nr_threads;
    ModuleBase* prev_module;
    MultiThreadsModuleConfig() {}
    MultiThreadsModuleConfig(int threads, ModuleBase* prev) : nr_threads(threads), prev_module(prev) {}
};

struct NNConfig {
    MultiThreadsModuleConfig basic_config;
    struct ModelConfig {
        //! add more model config
        std::string model_path;
        std::string engine_load_base;
        std::string engine_save_base;
        std::string precision;
        std::string input_names;
        std::string input_shapes;
        ModelConfig() {}
    };
    ModelConfig model_config;
    NNConfig() {}
    NNConfig(int threads) : basic_config(threads, nullptr) {}
    NNConfig(int threads, ModuleBase* prev) : basic_config(threads, prev) {}
};

} // namespace e2e_sample

#endif