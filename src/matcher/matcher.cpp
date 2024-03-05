/**
 ******************************************************************************
 * @file           : include/matcher/matcher.h
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : The key points matcher with onnx model
 * @attention      : None
 * @date           : 24-1-28
 ******************************************************************************
 */

#include "image_process.h"
#include "logger/logger.h"
#include "matcher/matcher.h"

Matcher::Matcher( unsigned int threads_num )
    : m_threads_num( threads_num )
{
  INFO( logger, "Matcher is being created" );
}

Matcher::~Matcher()
{
  INFO( logger, "Matcher is being destroyed" );

  for ( auto& name : m_vec_input_names )
  {
    delete[] name;
  }
  m_vec_input_names.clear();

  for ( auto& name : m_vec_output_names )
  {
    delete[] name;
  }
  m_vec_output_names.clear();
}

int Matcher::initOrtEnv( const Config& config )
{
  INFO( logger, "Initializing matcher ort env" );

  try
  {
    m_env             = Ort::Env( ORT_LOGGING_LEVEL_WARNING, "Matcher" );
    m_session_options = Ort::SessionOptions();

    if ( m_threads_num == 0 )
    {
      m_threads_num == std::thread::hardware_concurrency();
    }

    m_session_options.SetIntraOpNumThreads( m_threads_num );
    m_session_options.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_ALL );
    INFO( logger, "Using {0} threads, with graph optimization level {1}", m_threads_num, GraphOptimizationLevel::ORT_ENABLE_ALL );

    if ( config.device == "cuda" )
    {
      INFO( logger, "Using CUDA for matcher" );

      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id                 = 0;                              // 这行设置 CUDA 设备 ID 为 0，这意味着 ONNX Runtime 将在第一个 CUDA 设备（通常是第一个 GPU）上运行模型。
      cuda_options.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchDefault;  // 这行设置 cuDNN 卷积算法搜索策略为默认值。cuDNN 是 NVIDIA 的深度神经网络库，它包含了许多用于卷积的优化算法。
      cuda_options.gpu_mem_limit             = 0;                              // 这行设置 GPU 内存限制为 0，这意味着 ONNX Runtime 可以使用所有可用的 GPU 内存。
      cuda_options.arena_extend_strategy     = 1;                              // 这行设置内存分配策略为 1，这通常意味着 ONNX Runtime 将在需要更多内存时扩展内存池。
      cuda_options.do_copy_in_default_stream = 1;                              // 行设置在默认流中进行复制操作为 1，这意味着 ONNX Runtime 将在 CUDA 的默认流中进行数据复制操作。
      cuda_options.has_user_compute_stream   = 0;                              // 这行设置用户计算流为 0，这意味着 ONNX Runtime 将使用其自己的计算流，而不是用户提供的计算流。
      cuda_options.default_memory_arena_cfg  = nullptr;                        // 这行设置默认内存区配置为 nullptr，这意味着 ONNX Runtime 将使用默认的内存区配置。

      m_session_options.AppendExecutionProvider_CUDA( cuda_options );
      m_session_options.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_ALL );
    }

    INFO( logger, "Loading matcher model from {0}", config.matcher_path );

    m_uptr_session = std::make_unique<Ort::Session>( m_env, config.matcher_path.c_str(), m_session_options );

    INFO( logger, "Matcher model loaded" );
    extractNodesInfo( IO::INPUT, m_vec_input_names, m_vec_input_shapes, m_uptr_session, m_allocator );
    extractNodesInfo( IO::OUTPUT, m_vec_output_names, m_vec_output_shapes, m_uptr_session, m_allocator );

    INFO( logger, "Matcher ort env initialized" );
  }
  catch ( const std::exception& e )
  {
    INFO( logger, "Failed to initialize matcher ort env: {0}", e.what() );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


std::vector<cv::Point2f> Matcher::preProcess( std::vector<cv::Point2f> key_points, const int& height, const int& width )
{
  return normalizeKeyPoints( key_points, height, width );
}

int Matcher::inference( const Config& config, const std::vector<cv::Point2f> key_points_src, const std::vector<cv::Point2f> key_points_dst, const cv::Mat& descriptor_src, const cv::Mat& descriptor_dst )
{
  INFO( logger, "Matcher inference start" );

  try
  {
    m_vec_input_shapes[ 0 ] = { 1, static_cast<int>( key_points_src.size() ), 2 };
    m_vec_input_shapes[ 1 ] = { 1, static_cast<int>( key_points_dst.size() ), 2 };
    m_vec_input_shapes[ 2 ] = { 1, static_cast<int>( key_points_src.size() ), 256 };
    m_vec_input_shapes[ 3 ] = { 1, static_cast<int>( key_points_dst.size() ), 256 };

    INFO( logger, "Input shapes initialized" );
    auto   memory_info_handler = Ort::MemoryInfo::CreateCpu( OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU );
    float* key_points_src_data = new float[ key_points_src.size() * 2 ];
    float* key_points_dst_data = new float[ key_points_dst.size() * 2 ];

    for ( int i = 0; i < key_points_src.size(); ++i )
    {
      key_points_src_data[ i * 2 ]     = key_points_src[ i ].x;
      key_points_src_data[ i * 2 + 1 ] = key_points_src[ i ].y;
    }

    for ( int i = 0; i < key_points_dst.size(); ++i )
    {
      key_points_dst_data[ i * 2 ]     = key_points_dst[ i ].x;
      key_points_dst_data[ i * 2 + 1 ] = key_points_dst[ i ].y;
    }

    float* descriptor_src_data;
    if ( descriptor_src.isContinuous() )
    {
      descriptor_src_data = const_cast<float*>( descriptor_src.ptr<float>( 0 ) );
    }
    else
    {
      cv::Mat temp_descriptor = descriptor_src.clone();
      descriptor_src_data     = const_cast<float*>( descriptor_src.ptr<float>( 0 ) );
    }

    float* descriptor_dst_data;
    if ( descriptor_dst.isContinuous() )
    {
      descriptor_dst_data = const_cast<float*>( descriptor_dst.ptr<float>( 0 ) );
    }
    else
    {
      cv::Mat temp_descriptor = descriptor_dst.clone();
      descriptor_dst_data     = const_cast<float*>( descriptor_dst.ptr<float>( 0 ) );
    }

    INFO( logger, "Matcher inference input tensors created" );
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, key_points_src_data, key_points_src.size() * 2, m_vec_input_shapes[ 0 ].data(), m_vec_input_shapes[ 0 ].size() ) );
    input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, key_points_dst_data, key_points_dst.size() * 2, m_vec_input_shapes[ 1 ].data(), m_vec_input_shapes[ 1 ].size() ) );
    input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, descriptor_src_data, key_points_src.size() * 256, m_vec_input_shapes[ 2 ].data(), m_vec_input_shapes[ 2 ].size() ) );
    input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, descriptor_dst_data, key_points_dst.size() * 256, m_vec_input_shapes[ 3 ].data(), m_vec_input_shapes[ 3 ].size() ) );

    m_timer.tic();
    auto output_tensor_temp = m_uptr_session->Run( Ort::RunOptions{ nullptr }, m_vec_input_names.data(), input_tensors.data(), input_tensors.size(), m_vec_output_names.data(), m_vec_output_names.size() );
    INFO( logger, "matcher inference time consumed: {0}", m_timer.tocGetDuration() );

    m_vec_output_tensor = std::move( output_tensor_temp );
  }
  catch ( const std::exception& e )
  {
    ERROR( logger, "**** matcher inference failed with error message: {0} ****", e.what() );
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int Matcher::postProcess( const Config& config )
{
  INFO( logger, "Matcher post process start" );
  try
  {
    std::vector<int64_t> matches_shape = m_vec_output_tensor[ 0 ].GetTensorTypeAndShapeInfo().GetShape();
    int64_t*             matches_ptr   = (int64_t*)m_vec_output_tensor[ 0 ].GetTensorMutableData<void>();

    std::vector<int64_t> scores_shape = m_vec_output_tensor[ 1 ].GetTensorTypeAndShapeInfo().GetShape();
    float*               scores_ptr   = (float*)m_vec_output_tensor[ 1 ].GetTensorMutableData<void>();

    // create matches indices
    std::set<std::pair<int, int>> matches;
    int                           count = 0;
    for ( int i = 0; i < matches_shape[ 0 ] * 2; i += 2 )
    {
      if ( matches_ptr[ i ] > -1 && matches_ptr[ i + 1 ] > -1 && scores_ptr[ count ] > m_match_threshold_ )
      {
        matches.insert( std::make_pair( matches_ptr[ i ], matches_ptr[ i + 1 ] ) );
      }
      count++;
    }
    m_matched_indices = matches;
  }
  catch ( const std::exception& e )
  {
    ERROR( logger, "**** matcher post process failed with error message: {0} ****", e.what() );
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

std::set<std::pair<int, int>> Matcher::inferenceDescriptorPair( const Config& config, const std::vector<cv::Point2f> key_points_src, const std::vector<cv::Point2f> key_points_dst, const cv::Mat& descriptor_src, const cv::Mat& descriptor_dst )
{
  auto key_points_src_norm = preProcess( key_points_src, m_height, m_width );
  auto key_points_dst_norm = preProcess( key_points_dst, m_height, m_width );

  inference( config, key_points_src_norm, key_points_dst_norm, descriptor_src, descriptor_dst );
  postProcess( config );


  return m_matched_indices;
}

void Matcher::setScales( const std::vector<float>& scales )
{
  m_vec_sacles = scales;
}

void Matcher::setHeight( const int& height )
{
  m_height = height;
}

void Matcher::setWidth( const int& width )
{
  m_width = width;
}


void Matcher::setMatchThreshold( const float& threshold )
{
  m_match_threshold_ = threshold;
}

float Matcher::getMatchThreshold() const
{
  return m_match_threshold_;
}

void Matcher::setParams( const std::vector<float>& scales, const int& height, const int& width, const float& threshold )
{
  m_vec_sacles       = scales;
  m_height           = height;
  m_width            = width;
  m_match_threshold_ = threshold;
}
