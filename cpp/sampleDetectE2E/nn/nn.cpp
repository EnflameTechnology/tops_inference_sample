#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>

#include "TopsInference/TopsInferRuntime.h"

#include "utils/timer.h"

#include "nn.h"

#define ENFLAME_RUNTIME_CHECK(_expr)                                           \
    do {                                                                         \
        if (topsError_t::topsSuccess != _expr) {                                   \
            fprintf(stderr, "EnFlame Runtime ERROR : %s @ %s:%d\n", topsGetErrorString(_expr), __FILE__, __LINE__);\
            raise(SIGUSR1);                                                                \
        }                                                                          \
    } while (0)

#define ENFLAEM_TIF_CHECK(_expr) \
    do {                                                                         \
        if (TopsInference::TIFStatus::TIF_SUCCESS != _expr) {                                   \
            fprintf(stderr, "EnFlame TopsInference ERROR : %d @ %s:%d\n", _expr, __FILE__, __LINE__);\
            raise(SIGUSR1);                                                                \
        }                                                                          \
    } while (0)

#define TIF_ERROR_CHECK(_expr)  \
    if (!_expr) {\
        int32_t error_count = error_manager->getErrorCount();\
        for (int32_t i = 0; i < error_count; i++) {\
            const char* error_msg = error_manager->getErrorMsg(i);\
            TopsInference::TIFStatus error_status = error_manager->getErrorStatus(i);\
            fprintf(stderr, "EnFlame TopsInference ERROR : (%d)%s @ %s:%d\n", error_status, error_msg, __FILE__, __LINE__);\
        }\
        error_manager->clear();\
        state = STATE::ERROR;\
        break;\
    }\

namespace {
size_t get_dtype_size(TopsInference::DataType dtype) {
    switch(dtype) {
        case TopsInference::DataType::TIF_BOOL :
        case TopsInference::DataType::TIF_INT8 :
        case TopsInference::DataType::TIF_UINT8 : return 1;
        case TopsInference::DataType::TIF_INT16 :
        case TopsInference::DataType::TIF_UINT16 :
        case TopsInference::DataType::TIF_FP16 :
        case TopsInference::DataType::TIF_BF16 : return 2;
        default : return 4;
    }
}
}

namespace e2e_sample {

TopsDetect::TopsDetect(const NNConfig& config) : MultiThreadsWorker(config.basic_config.nr_threads), m_config(config) {
    m_nn_outputs = std::make_unique<ThreadSafeQueue<E2ENNOutput>>("nn_outputs");
    m_gcu_mem_pool = std::make_unique<GCUMemoryBufferPool>();
}

TopsDetect::~TopsDetect() {
    if (m_is_working) {
        stop();
    }
    MultiThreadsWorker::join_all();
    if (m_error_manager) {
        TopsInference::release_error_manager((TopsInference::IErrorManager*)(m_error_manager));
    }
    ENFLAEM_TIF_CHECK(TopsInference::topsInference_finish());
    m_gcu_mem_pool.reset();
    fprintf(stderr, "TopsDetect Module Deconstructor End.\n");
}

//! FIXME:目前推理结果并未做后处理，因此推理 output 并没有打包成 E2ENNOutput，显存也是复用的。如果有异步后处理，E2ENNOutput 需要处理。
void TopsDetect::work_func(int thread_id) {
    enum class STATE {
        START = 0,
        BUILD = 1,
        WORKING = 2,
        HOLD_ON = 3,
        ERROR = 4,
        QUIT = 5,
        STOP,
    };
    STATE state = STATE::START;

    //! resources
    ModuleBase *cv_module = this->m_config.basic_config.prev_module;
    TopsInference::IErrorManager *error_manager = (TopsInference::IErrorManager*)(this->m_error_manager);
    TopsInference::handler_t handle = nullptr;
    TopsInference::IEngine *engine = nullptr;
    TopsInference::IParser *parser = nullptr;
    TopsInference::IOptimizer *optimizer = nullptr;
    TopsInference::INetwork *network = nullptr;
    TopsInference::topsInferStream_t stream = nullptr;
    TopsInference::IFuture* future = nullptr;
    size_t nr_tasks = 0;

    //! tensor
    using TIF_Tensor = TopsInference::TensorPtr_t;
    std::vector<TIF_Tensor> input_tensor_list;
    std::vector<TIF_Tensor> output_tensor_list;
    std::vector<std::shared_ptr<MemoryObj>> output_buffers;

    //! path
    auto model_path = this->m_config.model_config.model_path;
    auto engine_path_load = this->m_config.model_config.engine_load_base + "-thread-" + std::to_string(thread_id) + ".bin";
    auto engine_path_save = this->m_config.model_config.engine_save_base + "-thread-" + std::to_string(thread_id) + ".bin";

    //! timer
    Timer timer;
    double total_time = 0;

    auto str2tif_precison = [](const std::string& precision) -> auto {
        if (strcmp(precision.c_str(), "default") == 0) {
            return TopsInference::BuildFlag::TIF_KTYPE_DEFAULT;
        }
        return TopsInference::BuildFlag::TIF_KTYPE_MIX_FP16;
    };

    auto release_all_resource = [&]() -> void {
        if (stream) {
            TopsInference::destroy_stream(stream);
        }
        if (future) {
            TopsInference::destroy_future(future);
        }
        for (auto&& tif_tensor : input_tensor_list) {
            if (tif_tensor) {
                TopsInference::destroy_tensor(tif_tensor);
            }
        }
        for (auto&& tif_tensor : output_tensor_list) {
            if (tif_tensor) {
                TopsInference::destroy_tensor(tif_tensor);
            }
        }
        if (network) {
            TopsInference::release_network(network);
            network = nullptr;
        }
        if (optimizer) {
            TopsInference::release_optimizer(optimizer);
            optimizer = nullptr;
        }
        if (parser) {
            TopsInference::release_parser(parser);
            parser = nullptr;
        }
        if (engine) {
            TopsInference::release_engine(engine);
            engine = nullptr;
        }
        if (handle) {
            TopsInference::release_device(handle);
            handle = nullptr;
        }
    };

    int32_t nr_inputs;
    int32_t nr_outputs;

    while (this->m_is_working) {
        switch(state) {
            case STATE::START : {
                std::unique_lock<std::mutex> lck(this->m_topsinfer_init_mutex);
                uint32_t cluster_ids[] = {(uint32_t)(thread_id % 2)};
                //! device id default 0
                handle = TopsInference::set_device(0, cluster_ids, 1, error_manager);
                TIF_ERROR_CHECK(handle)
                TopsInference::create_stream(&stream);
                state = STATE::BUILD;
                break;
            }
            case STATE::BUILD : {
                std::unique_lock<std::mutex> lck(this->m_topsinfer_init_mutex);
                //! try load engine first
                bool has_valid_engine = false;
                if (access(engine_path_load.c_str(), F_OK) == 0) {
                    engine = TopsInference::create_engine(error_manager);
                    TIF_ERROR_CHECK(engine)
                    has_valid_engine = engine->loadExecutable(engine_path_load.c_str());
                }
                if (has_valid_engine) {
                    fprintf(stderr, "Load Engine File : %s\n", engine_path_load.c_str());
                }
                else {
                    if (access(model_path.c_str(), F_OK) == 0) {
                        //! read from model and build engine
                        TopsInference::IParser *parser = TopsInference::create_parser(TopsInference::TIF_ONNX, error_manager);
                        TIF_ERROR_CHECK(parser)
                        auto input_names = this->m_config.model_config.input_names;
                        auto input_shapes = this->m_config.model_config.input_shapes;
                        if (!input_names.empty()) {
                                parser->setInputNames(input_names.c_str());
                        }
                        if (!input_shapes.empty()) {
                            parser->setInputShapes(input_shapes.c_str());
                        }
                        TopsInference::IOptimizer *optimizer = TopsInference::create_optimizer(error_manager);
                        TIF_ERROR_CHECK(optimizer)
                        TopsInference::INetwork *network = parser->readModel(model_path.c_str());
                        fprintf(stderr, "Reading Model File : %s\n", model_path.c_str());
                        TIF_ERROR_CHECK(network)
                        TopsInference::IOptimizerConfig *optimizer_config = optimizer->getConfig();
                        optimizer_config->setBuildFlag(str2tif_precison(this->m_config.model_config.precision));
                        if (engine) {
                            TopsInference::release_engine(engine);
                            engine = nullptr;
                        }
                        engine = optimizer->build(network);
                        if (!engine_path_save.empty()) {
                            engine->saveExecutable(engine_path_save.c_str());
                        }
                        TopsInference::release_network(network);
                        TopsInference::release_optimizer(optimizer);
                        TopsInference::release_parser(parser);
                        has_valid_engine = true;
                    }
                    else {
                        fprintf(stderr, "No such file : %s\n", model_path.c_str());
                        has_valid_engine = false;
                    }
                }
                if (has_valid_engine) {
                    //! create tensor for input and output
                    nr_inputs = engine->getInputNum();
                    nr_outputs = engine->getOutputNum();
                    input_tensor_list.resize(nr_inputs);        //! input tensors
                    output_tensor_list.resize(nr_outputs);      //! output tensors
                    output_buffers.resize(nr_outputs);          //! output device buffers
                    for (int i = 0; i < nr_inputs; ++i) {
                        input_tensor_list[i] = TopsInference::create_tensor();
                        input_tensor_list[i]->setDeviceType(TopsInference::DataDeviceType::DEVICE);
                        input_tensor_list[i]->setDims(engine->getInputShape(i));
                    }
                    for (int i = 0; i < nr_outputs; ++i) {
                        output_tensor_list[i] = TopsInference::create_tensor();
                        output_tensor_list[i]->setDeviceType(TopsInference::DataDeviceType::DEVICE);
                        auto output_dims = engine->getOutputShape(i);
                        auto output_dtype = engine->getOutputDataType(i);
                        size_t size_in_bytes = get_dtype_size(output_dtype);
                        for (int32_t j = 0; j < output_dims.nbDims; ++j) {
                            size_in_bytes *= output_dims.dimension[j];
                        }
                        output_buffers[i] = m_gcu_mem_pool->get_gcu_memory(size_in_bytes);
                        output_tensor_list[i]->setOpaque(output_buffers[i]->ptr);
                        output_tensor_list[i]->setDims(output_dims);
                    }
                    this->m_build_engine_done[thread_id].set_value();
                    state = STATE::WORKING;
                }
                else {
                    state = STATE::ERROR;
                }
                break;
            }
            case STATE::WORKING : {
                std::shared_ptr<E2EImage> resized_image = cv_module->get_task();
                if (resized_image) {
                    nr_tasks++;
                    // ! Do Inference
                    timer.start();
                    for (int i = 0; i < nr_inputs; ++i) {
                        input_tensor_list[i]->setOpaque(resized_image->buf->extend_data);
                    }
                    future = TopsInference::create_future();
                    auto ret = engine->runV2(input_tensor_list.data(), output_tensor_list.data(), stream, future);
                    future->wait();
                    TopsInference::destroy_future(future);
                    future = nullptr;
                    double infer_time = timer.elapsed();
                    total_time += infer_time;
                    float avg_fps = nr_tasks * 1000.0 / total_time;
                    fprintf(stderr, "{\"message\": \"e2e_metric\", \"ops\": \"infer\", \"value\": %.2f, \"thread_id\": %d, \"ret\": %d, \"avg_fps\": %.2f}\n", infer_time * 1000, thread_id, ret, avg_fps);
                    state = STATE::WORKING;
                }
                else {
                    state = STATE::HOLD_ON;
                }
                break;
            }
            case STATE::HOLD_ON : {
                usleep(100);
                state = STATE::WORKING;
                break;
            }
            case STATE::ERROR : {
                this->m_build_engine_done[thread_id].set_value();
                state = STATE::QUIT;
                break;
            }
            case STATE::QUIT : {
                this->m_is_working = false;
                state = STATE::STOP;
            }
            default : break;
        };
    }
    
    switch(state) {
        case STATE::START :
        case STATE::BUILD :
        case STATE::ERROR : {
            this->m_build_engine_done[thread_id].set_value();
            break;
        }
        default : break;
    }

    {
        std::unique_lock<std::mutex> lck(this->m_topsinfer_init_mutex);
        release_all_resource();
    }
    
    fprintf(stderr, "NN Worker %d Stop.\n", thread_id);
}

void TopsDetect::start() {
    ENFLAEM_TIF_CHECK(TopsInference::topsInference_init());
    m_error_manager = (void*)(TopsInference::create_error_manager());
    m_build_engine_done.resize(m_config.basic_config.nr_threads);
    MultiThreadsWorker::start();
    fprintf(stderr, "TopsDetect Start %d Threads.\n", m_config.basic_config.nr_threads);
    //! wait all threads load or build engine
    for (auto&& promise : m_build_engine_done) {
        promise.get_future().get();
    }
}

void TopsDetect::stop() {
    fprintf(stderr, "Try Stop TopsDetect Module.\n");
    m_is_working = false;
}

E2ENNOutput TopsDetect::get_output() {
    auto res = m_nn_outputs->get();
    return res.second;
}

}