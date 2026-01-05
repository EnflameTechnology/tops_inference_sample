#include <signal.h>
#include <sys/types.h>
#include <unistd.h>

#include <iostream>

#include "topscv/topscv.h"
#include "utils/image_utils.h"

#include "resize.h"

#define ENFLAME_RUNTIME_CHECK(_expr)                                           \
  do {                                                                         \
    if (topsError_t::topsSuccess != _expr) {                                   \
      fprintf(stderr, "EnFlame Runtime ERROR : %s @ %s:%d\n",                  \
              topsGetErrorString(_expr), __FILE__, __LINE__);                  \
      raise(SIGUSR1);                                                          \
    }                                                                          \
  } while (0)

#define TOPS_CV_CHECK(_expr)                                                   \
  do {                                                                         \
    if (topscvStatus_t::TOPSCV_STATUS_SUCCESS != _expr) {                      \
      const char *info = nullptr;                                              \
      topscvErrCode2String(_expr, &info);                                      \
      fprintf(stderr, "TopsCV ERROR : %s @ %s:%d\n", info, __FILE__,           \
              __LINE__);                                                       \
      raise(SIGUSR1);                                                          \
    }                                                                          \
  } while (0)

namespace e2e_sample {

TopsResize::TopsResize(const MultiThreadsModuleConfig &config) : MultiThreadsWorker(config.nr_threads), m_config(config) {
    m_resized_images = std::make_unique<ThreadSafeQueue<std::shared_ptr<E2EImage>>>("resized_images");
    m_images_pool = std::make_unique<ImageBufferPool>();
}

TopsResize::~TopsResize() {
    if (m_is_working) {
        stop();
    }
    MultiThreadsWorker::join_all();
    for (auto &&iter : m_rt_resource) {
        if (iter.stream) {
            topsStream_t stream = static_cast<topsStream_t>(iter.stream);
            ENFLAME_RUNTIME_CHECK(topsStreamDestroy(stream));
        }
        if (iter.start_event) {
            topsEvent_t event = static_cast<topsEvent_t>(iter.start_event);
            ENFLAME_RUNTIME_CHECK(topsEventDestroy(event));
        }
        if (iter.end_event) {
            topsEvent_t event = static_cast<topsEvent_t>(iter.end_event);
            ENFLAME_RUNTIME_CHECK(topsEventDestroy(event));
        }
    }
    //! release resources
    for (auto &&op : m_ops) {
        topscvOperator_t cv_op = static_cast<topscvOperator_t>(op);
        TOPS_CV_CHECK(topscvOperatorDestroy(&cv_op));
    }
    m_resized_images.reset();
    m_images_pool.reset();
    fprintf(stderr, "TopsResize Module Deconstructor End.\n");
}

void TopsResize::work_func(int thread_id) {
    ModuleBase *decoder = this->m_config.prev_module;
    //! rt resource
    topsStream_t stream = static_cast<topsStream_t>(this->m_rt_resource[thread_id].stream);
    topsEvent_t start_event = static_cast<topsEvent_t>(this->m_rt_resource[thread_id].start_event);
    topsEvent_t end_event = static_cast<topsEvent_t>(this->m_rt_resource[thread_id].end_event);
    topscvOperator_t op = static_cast<topscvOperator_t>(this->m_ops[thread_id]);
    //! layout
    static const size_t ow = 640;
    static const size_t oh = 640;
    topscvImage_t src_img = nullptr;
    topscvImage_t dst_img = nullptr;
    topscvResizeI_t resize_mode;
    resize_mode.width = ow;
    resize_mode.height = oh;
    resize_mode.interpolation_type = topscvInterpolationType_t::TOPSCV_INTERP_NEAREST;
    size_t nr_tasks = 0;
    float total_time = 0;
	//! work loop
    while (this->m_is_working) {
        auto decoded_image = decoder->get_task();
        if (decoded_image) {
            if (ow != decoded_image->image_info.width || oh != decoded_image->image_info.height) {
                //! do resize
                nr_tasks++;
                std::shared_ptr<E2EImage> resized_image = m_images_pool->get_gcu_image(ow, oh, ImageFormat::RGB24);
                src_img = (topscvImage_t)(decoded_image->buf->ptr);
                dst_img = (topscvImage_t)(resized_image->buf->ptr);
                ENFLAME_RUNTIME_CHECK(topsEventRecord(start_event, stream));
                TOPS_CV_CHECK(topscvResizeSubmit(op, stream, src_img, dst_img, &resize_mode, topscvBackend_t::TOPSCV_BACKEND_VPP));
                ENFLAME_RUNTIME_CHECK(topsEventRecord(end_event, stream));
                ENFLAME_RUNTIME_CHECK(topsEventSynchronize(end_event));
                float duration = 0;
                ENFLAME_RUNTIME_CHECK(topsEventElapsedTime(&duration, start_event, end_event));
                total_time += duration;
                float avg_fps = nr_tasks * 1000.0 / total_time;
                fprintf(stderr, "{\"message\": \"e2e_metric\", \"ops\": \"resize\", \"avg_fps\": %.2f, \"thread_id\": %d}\n", avg_fps, thread_id);
                this->m_resized_images->insert(resized_image);
            }
            else {
                //! skip resize
                this->m_resized_images->insert(decoded_image);
            }
        } else {
            usleep(100);
        }
    }
    ENFLAME_RUNTIME_CHECK(topsStreamSynchronize(stream));
    fprintf(stderr, "Resize Worker %d Stop.\n", thread_id);
}

void TopsResize::start() {
    m_rt_resource.resize(m_config.nr_threads);
    m_ops.resize(m_config.nr_threads);
  	for (int i = 0; i < m_config.nr_threads; ++i) {
		topsStream_t stream = nullptr;
    	ENFLAME_RUNTIME_CHECK(topsStreamCreate(&stream));
        topsEvent_t start_event = nullptr;
        topsEvent_t end_event = nullptr;
        ENFLAME_RUNTIME_CHECK(topsEventCreate(&start_event));
        ENFLAME_RUNTIME_CHECK(topsEventCreate(&end_event));
        m_rt_resource[i].stream = static_cast<void*>(stream);
        m_rt_resource[i].start_event = static_cast<void*>(start_event);
        m_rt_resource[i].end_event = static_cast<void*>(end_event);
    	topscvOperator_t op = nullptr;
    	TOPS_CV_CHECK(topscvResizeCreate(&op));
    	m_ops[i] = static_cast<void*>(op);
  	}
  	fprintf(stderr, "TopsResize Start %d Threads.\n", m_config.nr_threads);
  	MultiThreadsWorker::start();
}

void TopsResize::stop() {
  	fprintf(stderr, "Try Stop Resize Module.\n");
  	m_is_working = false;
}

std::shared_ptr<E2EImage> TopsResize::get_task() {
    auto res = m_resized_images->get();
    if (res.first) {
        return res.second;
    }
    else {
        return nullptr;
    }
}

} // namespace e2e_sample