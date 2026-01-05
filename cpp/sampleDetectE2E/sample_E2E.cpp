#include <atomic>
#include <future>
#include <fstream>
#include <iostream>
#include <memory>

#include <signal.h>
#include <unistd.h>

#include "utils/arg_parser.hpp"
#include "utils/json.hpp"
#include "utils/timer.h"

#include "decoder/ffmpeg_gcu.h"
#include "nn/nn.h"
#include "cv/resize.h"

#define ENFLAME_RUNTIME_CHECK(_expr)                                           \
  do {                                                                         \
    if (topsError_t::topsSuccess != _expr) {                                   \
      fprintf(stderr, "EnFlame Runtime ERROR : %s @ %s:%d\n", topsGetErrorString(_expr), __FILE__, __LINE__);\
      raise(SIGUSR1);                                                                \
    }                                                                          \
  } while (0)

#define ENFLAME_ASSERT(_expr, _fmt, ...)                                       \
  do {                                                                         \
    if (!_expr) {                                                              \
      fprintf(stderr, "[ENFLAME ASSERT : %s (%d)] :\n", __FILE__, __LINE__);   \
      fprintf(stderr, _fmt, ##__VA_ARGS__);                                    \
      fprintf(stderr, "\n");                                                   \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

namespace {
using FFmpegGCU = e2e_sample::FFmpegGCU;
using TopsResize = e2e_sample::TopsResize;
using TopsDetect = e2e_sample::TopsDetect;

struct E2EVideoJsonOneVideo {
    std::string _name;
    std::string _addr;
    int _fer;
    bool _reopen;
    int _repeat; 
};
void from_json(const nlohmann::json &json_obj, E2EVideoJsonOneVideo &t) {
    json_obj.at("name").get_to(t._name);
    json_obj.at("addr").get_to(t._addr);
    json_obj.at("frame_extraction_rate").get_to(t._fer);
    json_obj.at("reopen").get_to(t._reopen);
    json_obj.at("repeat").get_to(t._repeat);
}

struct E2EVideoJsonFile {
    std::vector<E2EVideoJsonOneVideo> _videos;
};
void from_json(const nlohmann::json &json_obj, E2EVideoJsonFile &t) {
    auto size = json_obj["videos"].size();
    t._videos.resize(size);
    for (size_t i = 0; i < size; ++i) {
        t._videos[i] = json_obj["videos"][i];
    }
}

struct E2EModelJsonFile {
    std::string _model_path;
    std::string _engine_load_base;
    std::string _engine_save_base;
    std::string _precision;
    std::string _input_names;
    std::string _input_shapes;
};
void from_json(const nlohmann::json &json_obj, E2EModelJsonFile &t) {
    json_obj.at("model_path").get_to(t._model_path);
    json_obj.at("engine_load_base").get_to(t._engine_load_base);
    json_obj.at("engine_save_base").get_to(t._engine_save_base);
    json_obj.at("precision").get_to(t._precision);
    json_obj.at("input_names").get_to(t._input_names);
    json_obj.at("input_shapes").get_to(t._input_shapes);
}

struct E2EModules {
    std::unique_ptr<FFmpegGCU> p_deocder = nullptr;
    std::unique_ptr<TopsResize> p_resize = nullptr;
    std::unique_ptr<TopsDetect> p_detect = nullptr;
    E2EModules() {}
};

//! global vars
E2EModules modules;
std::promise<void> quit_process;

//! handle signal 10 to stop
void signal_handler_fun(int signum) {
    fprintf(stderr, "Recieve Signal : %d\n", signum);

    if (modules.p_deocder) {
        modules.p_deocder->stop_all();
    }

    if (modules.p_resize) {
        modules.p_resize->stop();
    }
    
    if (modules.p_detect) {
        modules.p_detect->stop();
    }
    //! tell main process to quit
    quit_process.set_value();
}

}

int main(int argc, char** argv) {
    signal(SIGUSR1, signal_handler_fun);
    ap::parser p(argc, argv);
    p.add("-v", "--video_config", "Json file for each video path. Required", ap::mode::REQUIRED);
    p.add("-m", "--model_config", "model config file for NN inference", ap::mode::REQUIRED);
    p.add("-i", "--card_id", "Device ID. Default is 0.", ap::mode::OPTIONAL);
    p.add("-c", "--nr_cv_workers", "num of workers for CV module. Default is 1", ap::mode::OPTIONAL);
    p.add("-n", "--nr_nn_workers", "num of workers for NN module. Default is 1", ap::mode::OPTIONAL);
    
    auto args = p.parse();

    if (! args.parsed_successfully()) {
        fprintf(stderr, "Parse Fail. Use -h/--help for detail.\n");
        return 0;
    }

    auto card_id = 0;
    auto video_config_path = args["-v"];
    auto model_config_path = args["-m"];
    auto nr_cv_workers = 1;
    auto nr_nn_workers = 1;
    
    if (!args["-i"].empty()) {
        card_id = std::atoi(args["-i"].c_str());
    }
    if (!args["-c"].empty()) {
        nr_cv_workers = std::atoi(args["-c"].c_str());
    }
    if (!args["-n"].empty()) {
        nr_nn_workers = std::atoi(args["-n"].c_str());
    }
    //! set device
    ENFLAME_RUNTIME_CHECK(topsSetDevice(card_id));

    //! test D2H
    // {
    //     std::vector<void*> gcu_buffer{10, nullptr};
    //     std::vector<void*> cpu_buffer{10, nullptr};
    //     e2e_sample::Timer timer;
    //     double total_time = 0;
    //     size_t size_in_bytes = 3 * 1920 * 1080;
    //     for (int i = 0; i < 10; ++i) {
    //         ENFLAME_RUNTIME_CHECK(topsMalloc(&gcu_buffer[i], size_in_bytes));
    //         cpu_buffer[i] = new uint8_t[size_in_bytes];
    //     }
    //     for (int i = 0; i < 30000; ++i) {
    //         auto idx = i % 10;
    //         timer.start();
    //         ENFLAME_RUNTIME_CHECK(topsMemcpyDtoH(cpu_buffer[idx], gcu_buffer[idx], size_in_bytes));
    //         double duration = timer.elapsed();
    //         total_time += duration;
    //     }
    //     for (int i = 0; i < 10; ++i) {
    //         ENFLAME_RUNTIME_CHECK(topsFree(gcu_buffer[i]));
    //         delete [] (uint8_t*)(cpu_buffer[i]);
    //     }
    //     fprintf(stderr, "Total Transform Time : %lf\n", total_time);
    //     return 0;
    // }

    //! decode module
    {
        modules.p_deocder = std::make_unique<FFmpegGCU>();
    }
    //! cv module
    {
        e2e_sample::MultiThreadsModuleConfig resize_config{nr_cv_workers, modules.p_deocder.get()};
        modules.p_resize  = std::make_unique<TopsResize>(resize_config);
    }
    //! nn module
    {
        //! read video json file
        nlohmann::json json_obj;
        std::ifstream fmodel(model_config_path);
        ENFLAME_ASSERT(fmodel.is_open(), "%s %s\n", model_config_path.c_str(), strerror(errno));
        fmodel >> json_obj;
        E2EModelJsonFile model_json_file;
        from_json(json_obj, model_json_file);
        e2e_sample::NNConfig nn_config{nr_nn_workers, modules.p_resize.get()};
        nn_config.model_config.model_path = model_json_file._model_path;
        //! maybe empty
        nn_config.model_config.engine_load_base = model_json_file._engine_load_base;
        //! maybe empty
        nn_config.model_config.engine_save_base = model_json_file._engine_save_base;
        //! should be fp16_mix or fp32
        nn_config.model_config.precision = model_json_file._precision;
        //! maybe empty
        nn_config.model_config.input_names = model_json_file._input_names;
        //! maybe empty
        nn_config.model_config.input_shapes = model_json_file._input_shapes;
        modules.p_detect = std::make_unique<TopsDetect>(nn_config);
    }

    //! start nn and cv module
    modules.p_detect->start();
    modules.p_resize->start();

    //! register all decoders
    {
        nlohmann::json json_obj;
        std::ifstream fvideo(video_config_path);
        ENFLAME_ASSERT(fvideo.is_open(), "%s %s\n", video_config_path.c_str(), strerror(errno));
        fvideo >> json_obj;
        E2EVideoJsonFile video_json_file;
        from_json(json_obj, video_json_file);
        for(auto&& video : video_json_file._videos) {
            e2e_sample::FFmpegGCUConfig config;
            config.name = video._name;
            config.addr = video._addr;
            config.card_id = card_id;
            config.fer = video._fer;
            config.reopen = video._reopen;
            int repeat = video._repeat;
            for (int i = 0; i < repeat; ++i) {
                config.dev_id = i % 8;
                modules.p_deocder->open_stream(config);
            }
        }
    }
    
    //! wait promise
    quit_process.get_future().wait();
    modules.p_detect.reset();
    modules.p_resize.reset();
    modules.p_deocder.reset();
    return 0;
}