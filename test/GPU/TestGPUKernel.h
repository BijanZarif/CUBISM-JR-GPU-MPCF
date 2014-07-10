// Kernel Test class

#include "GPU.h"

class TestGPUKernel
{
    public:
        inline void run()
        {
            GPU::bind_textures();
            GPU::TestKernel();
            GPU::unbind_textures();
        }
};
