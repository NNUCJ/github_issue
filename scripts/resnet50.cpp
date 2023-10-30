#include "arm_compute/runtime/XPU/XPUFunctions.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Allocator.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"
#include "arm_compute/runtime/PoolManager.h"
#include "utils/Utils.h"
#include "utils.h"

#include <vector>
#include <iostream>

using namespace arm_compute;
using namespace std;
using namespace utils;


class Xpu_Resnet50: public Example
{
public:

    bool do_setup(int argc, char **argv) override{
        NPYLoader npy ;
        NPYLoader conv1_weight_npy;
        std::ifstream stream;
        arm_COMPUTE_UNUSED(argc);
        arm_COMPUTE_UNUSED(argv);

        auto lifetime_mgr0 = std::make_shared<BlobLifetimeManager>();                           //Create lifetime manager for layer 
        auto lifetime_mgr1 = std::make_shared<BlobLifetimeManager>();                           //Create lifetime manager for tensor
        auto pool_mgr0 = std::make_shared<PoolManager>();
        auto pool_mgr1 = std::make_shared<PoolManager>();
        auto mm_layers = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr0, pool_mgr0);     // Create the memory manager
        auto mm_transitions = std::make_shared<MemoryManagerOnDemand>(lifetime_mgr1, pool_mgr1);     // Create the memory manager
        
        // Set memory manager where allowed to manage internal memory requirements
        conv1 = std::make_unique<XPUConvolutionLayer>(mm_layers); //  conv 
        layer1_0_conv1 = std::make_unique<XPUConvolutionLayer>(mm_layers); //  conv 
        layer1_0_conv2 = std::make_unique<XPUConvolutionLayer>(mm_layers); //  conv 
        layer1_0_conv3 = std::make_unique<XPUConvolutionLayer>(mm_layers); //  conv 
        layer1_0_downsample_conv = std::make_unique<XPUConvolutionLayer>(mm_layers);


        constexpr auto data_layout = DataLayout::NCHW;
        if(argc >1){
            stream.open(argv[1], std::fstream::in);
            if(stream.good()){
                load_npy_weight(src, argv[1], data_layout);
            }
            if(argc > 2){
                weight_root = argv[2];
                if(argc > 3){
                    output_filename = argv[3];
                }else {
                    output_filename = "/workspace/mlc/armdnn_self/data/armdnn_res/tmp_out.npy";
                }
            }  
        }else{
           std::cout << "./test_resnet  Input data required !"<<std::endl;
           exit(0);
        }
        // load input tensor 
        load_npy_weight(conv1_weight, weight_root + "/conv1_weight.npy", data_layout);
        
        //  -----------------------------------   load layer 0  
        // 1. --------------------------------- [start] bottleneck 0
        load_npy_weight(layer1_0_conv1_weight, weight_root + "/layer1_0_conv1_weight.npy", data_layout);
        load_npy_weight(layer1_0_conv2_weight, weight_root + "/layer1_0_conv2_weight.npy", data_layout);
        load_npy_weight(layer1_0_conv3_weight, weight_root + "/layer1_0_conv3_weight.npy", data_layout);
        load_npy_weight(layer1_0_downsample_conv_weight, weight_root + "/layer1_0_downsample_0_weight.npy", data_layout);
        // 1. --------------------------------- [end] bottleneck 1
        
       
        const TensorShape biases_shape_conv(conv1_weight.info()->dimension(3));
        const TensorShape out_shape_conv1(src.info()->dimension(0) /2 , src.info()->dimension(1) / 2, conv1_weight.info()->dimension(3));
        const TensorShape pool1_out_shape(out_shape_conv1.x()/2, out_shape_conv1.y()/2, out_shape_conv1.z());
        
       
        //  resnet50 conv1 + bn1 + relu + maxpooling2d1 
        conv1_biases.allocator()->init(TensorInfo(biases_shape_conv, 1, DataType::F32, data_layout));
        conv1_out.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32, data_layout));
        bn1_out.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32, data_layout)); 
        act1_out.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32, data_layout));
        pool1_out.allocator()->init(TensorInfo(pool1_out_shape, 1, DataType::F32, data_layout));
        pool1_copy.allocator()->init(TensorInfo(pool1_out_shape, 1, DataType::F32, data_layout));
        // layer(0)
        const TensorShape layer1_0_conv3_out_shape(pool1_out_shape.x(), pool1_out_shape.y(), pool1_out_shape.z() * 4);
        layer1_0_downsample_conv_out.allocator()->init(TensorInfo(layer1_0_conv3_out_shape, 1,  DataType::F32, data_layout));
        layer1_0_downsample_bn_out.allocator()->init(TensorInfo(layer1_0_conv3_out_shape, 1,  DataType::F32, data_layout));
        
        layer1_0_conv1_out.allocator()->init(TensorInfo(pool1_out_shape, 1,  DataType::F32, data_layout));
        layer1_0_bn1_out.allocator()->init(TensorInfo(pool1_out_shape, 1,  DataType::F32, data_layout));
        layer1_0_act1_out.allocator()->init(TensorInfo(pool1_out_shape, 1,  DataType::F32, data_layout));

        layer1_0_conv2_out.allocator()->init(TensorInfo(pool1_out_shape, 1,  DataType::F32, data_layout));
        layer1_0_bn2_out.allocator()->init(TensorInfo(pool1_out_shape, 1,  DataType::F32, data_layout));
        layer1_0_act2_out.allocator()->init(TensorInfo(pool1_out_shape, 1,  DataType::F32, data_layout));

        
        layer1_0_conv3_out.allocator()->init(TensorInfo(layer1_0_conv3_out_shape, 1,  DataType::F32, data_layout));
        layer1_0_bn3_out.allocator()->init(TensorInfo(layer1_0_conv3_out_shape, 1,  DataType::F32, data_layout));

        // layer1_0_downsample_conv_out.allocator()->init(TensorInfo(layer1_0_conv3_out_shape, 1,  DataType::F32, data_layout));
        // layer1_0_downsample_bn_out.allocator()->init(TensorInfo(layer1_0_conv3_out_shape, 1,  DataType::F32, data_layout));
        

        /* -----------------------End: [Initialize tensors] */

        /* [Configure functions] out_h==in_h out_w==in_w*/  
        conv1->configure(&src, &conv1_weight, nullptr, &conv1_out, PadStrideInfo(2 /* stride_x */, 2 /* stride_y */, 3/* pad_x */, 3 /* pad_y */));

        // bn1.configure(&conv1_out, &bn1_out, &mean, &var, &beta, &gamma, 0.00001, ActivationLayerInfo());
        BatchNoram2d_Config(bn1, conv1_out, bn1_out, bn1_mean, bn1_var, bn1_beta, bn1_gamma,
                            weight_root + "/bn1_running_mean.npy", 
                            weight_root + "/bn1_running_var.npy", 
                            weight_root + "/bn1_bias.npy", 
                            weight_root + "/bn1_weight.npy", 0.00001, data_layout, ActivationLayerInfo());

        act1.configure(&bn1_out, &act1_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        pool1.configure(&act1_out, &pool1_out, PoolingLayerInfo(PoolingType::MAX, 3, data_layout, PadStrideInfo(2, 2, 1, 1, DimensionRoundingType::FLOOR)));
        copy_pool.configure(&pool1_out, &pool1_copy);
        
        // --------------------------------- layer 0 
        // --------------------------------- [start]  Bottleneck d0

        layer1_0_downsample_conv->configure(&pool1_out, &layer1_0_downsample_conv_weight, nullptr, &layer1_0_downsample_conv_out, PadStrideInfo(1 , 1 , 0, 0));
        BatchNoram2d_Config(layer1_0_downsample_bn, layer1_0_downsample_conv_out,layer1_0_downsample_bn_out, 
                            layer1_0_downsample_bn_mean, layer1_0_downsample_bn_var, layer1_0_downsample_bn_beta, layer1_0_downsample_bn_gamma, 
                            weight_root+ "/layer1_0_downsample_1_running_mean.npy",weight_root+ "/layer1_0_downsample_1_running_var.npy", 
                            weight_root+ "/layer1_0_downsample_1_bias.npy", weight_root+ "/layer1_0_downsample_1_weight.npy", 
                            0.00001, data_layout, ActivationLayerInfo());

                            
        layer1_0_conv1->configure(&pool1_copy, &layer1_0_conv1_weight, nullptr, &layer1_0_conv1_out, PadStrideInfo(1 , 1 , 0, 0));
        BatchNoram2d_Config(layer1_0_bn1, layer1_0_conv1_out, layer1_0_bn1_out, 
                            layer1_0_bn1_mean, layer1_0_bn1_var, layer1_0_bn1_beta, layer1_0_bn1_gamma, 
                            weight_root+ "/layer1_0_bn1_running_mean.npy",weight_root+ "/layer1_0_bn1_running_var.npy", 
                            weight_root+ "/layer1_0_bn1_bias.npy", weight_root+ "/layer1_0_bn1_weight.npy", 
                            0.00001, data_layout, ActivationLayerInfo());
        layer1_0_act1.configure(&layer1_0_bn1_out, &layer1_0_act1_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        layer1_0_conv2->configure(&layer1_0_act1_out, &layer1_0_conv2_weight, nullptr, &layer1_0_conv2_out, PadStrideInfo(1 , 1 , 1, 1));
        BatchNoram2d_Config(layer1_0_bn2, layer1_0_conv2_out, layer1_0_bn2_out, 
                            layer1_0_bn2_mean, layer1_0_bn2_var, layer1_0_bn2_beta, layer1_0_bn2_gamma, 
                            weight_root+ "/layer1_0_bn2_running_mean.npy",weight_root+ "/layer1_0_bn2_running_var.npy", 
                            weight_root+ "/layer1_0_bn2_bias.npy", weight_root+ "/layer1_0_bn2_weight.npy", 
                            0.00001, data_layout, ActivationLayerInfo());
        layer1_0_act2.configure(&layer1_0_bn2_out, &layer1_0_act2_out, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        layer1_0_conv3->configure(&layer1_0_act2_out, &layer1_0_conv3_weight, nullptr, &layer1_0_conv3_out, PadStrideInfo(1 , 1 , 0, 0));
        BatchNoram2d_Config(layer1_0_bn3, layer1_0_conv3_out, layer1_0_bn3_out, 
                            layer1_0_bn3_mean, layer1_0_bn3_var, layer1_0_bn3_beta, layer1_0_bn3_gamma, 
                            weight_root+ "/layer1_0_bn3_running_mean.npy",weight_root+ "/layer1_0_bn3_running_var.npy", 
                            weight_root+ "/layer1_0_bn3_bias.npy", weight_root+ "/layer1_0_bn3_weight.npy", 
                            0.00001, data_layout, ActivationLayerInfo());
                            
        // ------------------------End: [Bottleneck 0  end]
        
        
        /* -----------------------End: [Configure functions] */


        /*[ Add tensors to memory manager ]*/
        memory_group0 = std::make_unique<MemoryGroup>(mm_transitions);
        // memory_group0 = std::make_unique<MemoryGroup>(mm_transitions);
        
        memory_group0->manage(&conv1_out);
        conv1_out.allocator()->allocate();
        memory_group0->manage(&bn1_out);
        bn1_out.allocator()->allocate();
        memory_group0->manage(&act1_out);
        act1_out.allocator()->allocate();
        memory_group0->manage(&pool1_out);
        pool1_out.allocator()->allocate();
        
        // ------------------------------ layer 0 ---------------------------------

        // Bottleneck 0 
        memory_group0->manage(&layer1_0_downsample_conv_out);
        layer1_0_downsample_conv_out.allocator()->allocate();
        memory_group0->manage(&layer1_0_downsample_bn_out);
        layer1_0_downsample_bn_out.allocator()->allocate();

        memory_group0->manage(&layer1_0_conv1_out);
        layer1_0_conv1_out.allocator()->allocate();
        memory_group0->manage(&layer1_0_bn1_out);
        layer1_0_bn1_out.allocator()->allocate();
        memory_group0->manage(&layer1_0_act1_out);
        layer1_0_act1_out.allocator()->allocate();
        memory_group0->manage(&layer1_0_conv2_out);
        layer1_0_conv2_out.allocator()->allocate();
        memory_group0->manage(&layer1_0_bn2_out);
        layer1_0_bn2_out.allocator()->allocate();
        memory_group0->manage(&layer1_0_act2_out);
        layer1_0_act2_out.allocator()->allocate();
        memory_group0->manage(&layer1_0_conv3_out);
        layer1_0_conv3_out.allocator()->allocate();
        memory_group0->manage(&layer1_0_bn3_out);
        layer1_0_bn3_out.allocator()->allocate();
        
        
        // Bottleneck 0  end
        // ------------------------------ layer 0 end ---------------------------------
        pool1_copy.allocator()->allocate();
        mm_layers->populate(allocator, 1);
        mm_transitions->populate(allocator, 2);
        
        return true; 
    }
    
    void do_run() override{
        memory_group0->acquire();
        // memory_group0->acquire();

        conv1->run();
        // save_to_npy(conv1_out, "/workspace/mlc/armdnn_self/data/armdnn_res_new/conv1_out.npy", is_fortran);
        bn1.run();
        // save_to_npy(bn1_out, "/workspace/mlc/armdnn_self/data/armdnn_res_new/bn1_out.npy", is_fortran);
        act1.run();
        // save_to_npy(act1_out, "/workspace/mlc/armdnn_self/data/armdnn_res_new/act1_out.npy", is_fortran);
        pool1.run();
        save_to_npy(pool1_out, "/workspace/mlc/armdnn_self/data/armdnn_res_new/maxpool.npy", is_fortran);
        copy_pool.run();
        layer1_0_downsample_conv->run();
        save_to_npy(layer1_0_downsample_conv_out, "/workspace/mlc/armdnn_self/data/armdnn_res_new/layer1_0_downsample_conv_out.npy", is_fortran);
        layer1_0_downsample_bn.run();
        save_to_npy(layer1_0_downsample_bn_out, "/workspace/mlc/armdnn_self/data/armdnn_res_new/layer1_0_downsample_bn_out.npy", is_fortran);

        layer1_0_conv1->run();
        save_to_npy(layer1_0_conv1_out, "/workspace/mlc/armdnn_self/data/armdnn_res_new/layer1_0_conv1_out.npy", is_fortran);
        layer1_0_bn1.run();
        save_to_npy(layer1_0_bn1_out, "/workspace/mlc/armdnn_self/data/armdnn_res_new/layer1_0_bn1_out.npy", is_fortran);
        layer1_0_act1.run();
        save_to_npy(layer1_0_act1_out, "/workspace/mlc/armdnn_self/data/armdnn_res_new/layer1_0_act1_out.npy", is_fortran);
        layer1_0_conv2->run();
        save_to_npy(layer1_0_conv2_out, "/workspace/mlc/armdnn_self/data/armdnn_res_new/layer1_0_conv2_out.npy", is_fortran);
        layer1_0_bn2.run();
        save_to_npy(layer1_0_bn2_out, "/workspace/mlc/armdnn_self/data/armdnn_res_new/layer1_0_bn2_out.npy", is_fortran);
        layer1_0_act2.run();
        save_to_npy(layer1_0_act2_out, "/workspace/mlc/armdnn_self/data/armdnn_res_new/layer1_0_act2_out.npy", is_fortran);
        layer1_0_conv3->run();
        save_to_npy(layer1_0_conv3_out, "/workspace/mlc/armdnn_self/data/armdnn_res_new/layer1_0_conv3_out.npy", is_fortran);
        layer1_0_bn3.run();
        save_to_npy(layer1_0_bn3_out, "/workspace/mlc/armdnn_self/data/armdnn_res_new/layer1_0_bn3.npy", is_fortran);

        memory_group0->release();
        // memory_group0->release();
        
    }


private:
    
    Tensor src{}; // input_tensor
    

    // 1. Initialize input and output tensor
    Tensor conv1_out{}, bn1_out{}, act1_out{}, pool1_out{}, pool1_copy{};
    // #################### Layer1 Sequential out tensors  #########################
    Tensor layer1_0_conv1_out{}, layer1_0_conv2_out{}, layer1_0_conv3_out{}, layer1_0_downsample_conv_out{};
    Tensor layer1_0_bn1_out{}, layer1_0_bn2_out{}, layer1_0_bn3_out{}, layer1_0_downsample_bn_out{};
    Tensor layer1_0_act1_out{}, layer1_0_act2_out{}, layer1_0_act3_out{};
    
    //  2. Initialize weight tensor
    Tensor conv1_weight{}, conv1_biases{} ; 
    Tensor bn1_mean{}, bn1_var{}, bn1_beta{}, bn1_gamma{};
    // #################### Layer1 Sequential weights #########################
    Tensor layer1_0_conv1_weight{},  layer1_0_conv2_weight{}, layer1_0_conv3_weight{}, layer1_0_downsample_conv_weight{};
    // Tensor layer1_0_conv1_biases{}, layer1_0_conv2_biases{}, 
    Tensor layer1_0_bn1_mean{}, layer1_0_bn1_var{}, layer1_0_bn1_beta{}, layer1_0_bn1_gamma{};
    Tensor layer1_0_bn2_mean{}, layer1_0_bn2_var{}, layer1_0_bn2_beta{}, layer1_0_bn2_gamma{};
    Tensor layer1_0_bn3_mean{}, layer1_0_bn3_var{}, layer1_0_bn3_beta{}, layer1_0_bn3_gamma{};
    Tensor layer1_0_downsample_bn_mean{}, layer1_0_downsample_bn_var{}, layer1_0_downsample_bn_beta{}, layer1_0_downsample_bn_gamma{};

   

    Allocator allocator{};
    
    // memory group
    std::unique_ptr<MemoryGroup> memory_group0;
    // std::unique_ptr<MemoryGroup> memory_group0;

    // Layers 
    std::unique_ptr<XPUConvolutionLayer> conv1{};
    XPUBatchNormalizationLayer bn1{};
    XPUActivationLayer act1{};
    XPUPoolingLayer pool1{};

    XPUCopy copy_pool;
    // #################### Layer1 Sequential #########################
    // Bottleneck 0  
    std::unique_ptr<XPUConvolutionLayer> layer1_0_conv1{};
    std::unique_ptr<XPUConvolutionLayer> layer1_0_conv2{};
    std::unique_ptr<XPUConvolutionLayer> layer1_0_conv3{};
    std::unique_ptr<XPUConvolutionLayer> layer1_0_downsample_conv{};
    XPUBatchNormalizationLayer layer1_0_bn1{};
    XPUBatchNormalizationLayer layer1_0_bn2{};
    XPUBatchNormalizationLayer layer1_0_bn3{};
    XPUBatchNormalizationLayer layer1_0_downsample_bn{};
    XPUActivationLayer layer1_0_act1{};
    XPUActivationLayer layer1_0_act2{};
    XPUActivationLayer layer1_0_act3{};
    
    
    bool is_fortran= false;
    std::string output_filename{};
    std::string weight_root{};
     
};

/**

    ./examples/xpu_resnet50  /workspace/mlc/armdnn_self/data/cat_numpy.npy /workspace/mlc/armdnn_self/data/resnet50_weights_npy
 */
int main(int argc, char **argvs)
{
    return utils::run_example<Xpu_Resnet50>(argc, argvs);
}