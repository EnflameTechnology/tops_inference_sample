#ifndef __BASIC_TYPE_H__
#define __BASIC_TYPE_H__

namespace e2e_sample {

enum class MemoryType {
    CPU_AV_FRAME = 0,
    GCU_CV_IMAGE = 1,
    GCU_MEM = 2,
};

struct MemoryObj {
    MemoryType mem_type;
    //! raw object pointer(*av_frame or topscvImage_t)
    void* ptr = nullptr;
    size_t size = 0;
    //! extend data by raw object if neccessary
    void* extend_data = nullptr;
    MemoryObj(MemoryType t) : mem_type(t) {}
    MemoryObj(MemoryType t, void* p) : mem_type(t), ptr(p) {}
    MemoryObj(MemoryType t, void* p, size_t s) : mem_type(t), ptr(p), size(s) {}
};

enum class ImageFormat {
    UNKNOW = -1,
    RGB24 = 0,
    BGR24 = 1,
    YUV420P = 2,
    YUV444P = 3
};

struct ImageInfo {
    size_t width;
    size_t height;
    ImageFormat format;
    ImageInfo(size_t w, size_t h, ImageFormat f) : width(w), height(h), format(f) {}
    ImageInfo(const ImageInfo& a) {
        width = a.width;
        height = a.height;
        format = a.format;
    }
    bool operator==(const ImageInfo& a) {
        return a.width == width && a.height == height && a.format == format;
    }
};

struct E2EImage {
    ImageInfo image_info;
    size_t frame_id{0};
    MemoryObj* buf{nullptr};
    E2EImage(size_t w, size_t h, ImageFormat f) : image_info(w, h, f) {}
};

}

#endif
