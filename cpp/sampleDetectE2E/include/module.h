#ifndef __MODULE_H__
#define __MODULE_H__

#include <memory>

#include "include/basic_type.h"

namespace e2e_sample {

class ModuleBase {
public:
    virtual std::shared_ptr<E2EImage> get_task() = 0;
protected:
    virtual ~ModuleBase() {}
};

}

#endif
