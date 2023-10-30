/*
 * Copyright (c) 2023 arm Limited
 * @Description: 
 * @Author: Paul Cheng(成杰)
 * @version: V0.1
 * @Date: 2023-09-22 06:47:08
 * @LastEditors: Paul Cheng(成杰)
 * @FilePath: /ComputeLibrary/examples/neon_test_conv.cpp
 * @brief brief description: Test armDNN conv2d, Sample Test Command:
 * *******./test_nchw_conv2d [data.npy]  [weight.npy]
 *  
 */
#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"
#include "utils/Utils.h"

#include <vector>
#include <iostream>

using namespace arm_compute;
using namespace std;
using namespace utils;


class NECon2dExample_nchw: public Example
{
public:

    bool do_setup(int argc, char **argv) override{
        NPYLoader npy ;
        std::ifstream stream;
        ARM_COMPUTE_UNUSED(argc);
        ARM_COMPUTE_UNUSED(argv);

        auto lifetime_mgr0 = std::make_shared<BlobLifetimeManager>();                           //Create lifetime manager for layer 
        auto lifetime_mgr1 = std::make_shared<BlobLifetimeManager>();                           //Create lifetime manager for tensor
        auto pool_mgr0 = std::make_shared<PoolManager>();
        auto pool_mgr1 = std::make_shared<PoolManager>();
        auto mm_layers = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr0, pool_mgr0);     // Create the memory manager
        auto mm_transitions = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr1, pool_mgr1);     // Create the memory manager
        
        // Set memory manager where allowed to manage internal memory requirements
        conv = std::make_unique<NEConvolutionLayer>(mm_layers); //  conv 

        constexpr auto data_layout = DataLayout::NCHW;
        if(argc >1){
            stream.open(argv[1], std::fstream::in);
            if(stream.good()){
                npy.open(argv[1], data_layout);
                npy.init_tensor(src, DataType::F32);
            }
            if(argc > 2){
                load_npy_weight(weight, argv[2]);
            }
        }else{
            //Initialize src tensor 
            constexpr unsigned int width_src_image  = 32;
            constexpr unsigned int height_src_image = 32;
            constexpr unsigned int ifm_src_img      = 1;
            const TensorShape src_shape(width_src_image, height_src_image, ifm_src_img);
            src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32, data_layout));

             // Initialize weight tensor
            constexpr unsigned int kernel_x_conv = 3;
            constexpr unsigned int kernel_y_conv = 3;
            constexpr unsigned int ofm_conv = 64;
            
            const TensorShape weight_shape_conv(kernel_x_conv, kernel_y_conv,  src.info()->dimension(2), ofm_conv);
            weight.allocator()->init(TensorInfo(weight_shape_conv, 1, DataType::F32));

        }


        const TensorShape out_shape_conv(src.info()->dimension(0), src.info()->dimension(1), weight.info()->dimension(3));
       
        out.allocator()->init(TensorInfo(out_shape_conv, 1, DataType::F32));
        
        /* -----------------------End: [Initialize tensors] */

        /* [Configure functions] out_h==in_h out_w==in_w*/  
        conv->configure(&src, &weight, nullptr, &out, PadStrideInfo(1 /* stride_x */, 1 /* stride_y */, 1/* pad_x */, 1 /* pad_y */));
        // conv->configure(&src, &weight, &biases, &out, PadStrideInfo(1 /* stride_x */, 1 /* stride_y */, 2/* pad_x */, 2 /* pad_y */));
        /* -----------------------End: [Configure functions] */


        /*[ Add tensors to memory manager ]*/
        memory_group0 = std::make_unique<MemoryGroup>(mm_transitions);
        memory_group1 = std::make_unique<MemoryGroup>(mm_transitions);
        
        memory_group0->manage(&out);
        out.allocator()->allocate();

        // Assigning a value to Tensor
        src.allocator()->allocate();
        if(npy.is_open()){
            npy.fill_tensor(src);
           
        }else{
            fill_random_tensor(src, -1.f, 1.f);
        }
        // print_tensor(src, data_layout);

        if (argc < 2){
            weight.allocator()->allocate();
            fill_tensor_value<float, Tensor>(weight, 0.5f);
        }
       
        // 
        mm_layers->populate(allocator, 1);
        mm_transitions->populate(allocator, 2);
        return true; 
    }
    
    void do_run() override{
        memory_group0->acquire();
    
        conv->run();
        save_to_npy(out, "./tmp.np", false);
        // print_tensor(out, DataLayout::NCHW);
        // release memory
        memory_group0->release();
        
    }

private:
    
    Tensor src{}; 
    Tensor weight{};
    Tensor biases{};
    Tensor out{};
    
    Allocator allocator{};
    
    // memory group
    std::unique_ptr<MemoryGroup> memory_group0{};
    std::unique_ptr<MemoryGroup> memory_group1{};

    // Layers 
    std::unique_ptr<NEConvolutionLayer> conv{};

     
};


int main(int argc, char **argvs)
{
    return utils::run_example<NECon2dExample_nchw>(argc, argvs);
}