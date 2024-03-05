/**
 ******************************************************************************
 * @file           : src/decoupled_onnx_runner.cpp
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-20
 ******************************************************************************
 */

#include "decoupled_onnx_runner/decoupled_onxx_runner.h"
#include "logger/logger.h"

DecoupledOnnxRunner::DecoupledOnnxRunner( unsigned int threads_num ) : threads_num_{ threads_num }
{
  INFO( logger, "DecoupledOnnxRunner created" );
}

DecoupledOnnxRunner::~DecoupledOnnxRunner()
{
  INFO( logger, "DecoupledOnnxRunner destroyed" );

  for ( auto& name : input_node_names_extractor_ )
  {
    delete[] name;
  }
  input_node_names_extractor_.clear();

  for ( auto& name : output_node_names_extractor_ )
  {
    delete[] name;
  }
  output_node_names_extractor_.clear();
}

int DecoupledOnnxRunner::initOrtEnv( const Config& config )
{
  INFO( logger, "initializing Ort Env" );

  try
  {
    env_extractor_ = Ort::Env( ORT_LOGGING_LEVEL_WARNING, "DecoupledOnnxRunner Extractor" );
    env_matcher_   = Ort::Env( ORT_LOGGING_LEVEL_WARNING, "DecoupledOnnxRunner Matcher" );

    // create session options
    session_options_extractor_ = Ort::SessionOptions();
    session_options_matcher_   = Ort::SessionOptions();

    if ( threads_num_ == 0 )
    {
      threads_num_ = std::thread::hardware_concurrency();
    }

    session_options_extractor_.SetIntraOpNumThreads( threads_num_ );
    session_options_extractor_.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_ALL );
    session_options_matcher_.SetIntraOpNumThreads( threads_num_ );
    session_options_matcher_.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_ALL );
    INFO( logger, "Using {0} threads, with graph optimization level {1}", threads_num_, GraphOptimizationLevel::ORT_ENABLE_ALL );

    if ( config.device == "cuda" )
    {
      INFO( logger, "Using CUDA provider with default options" );

      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id                 = 0;                              // 这行设置 CUDA 设备 ID 为 0，这意味着 ONNX Runtime 将在第一个 CUDA 设备（通常是第一个 GPU）上运行模型。
      cuda_options.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchDefault;  // 这行设置 cuDNN 卷积算法搜索策略为默认值。cuDNN 是 NVIDIA 的深度神经网络库，它包含了许多用于卷积的优化算法。
      cuda_options.gpu_mem_limit             = 0;                              // 这行设置 GPU 内存限制为 0，这意味着 ONNX Runtime 可以使用所有可用的 GPU 内存。
      cuda_options.arena_extend_strategy     = 1;                              // 这行设置内存分配策略为 1，这通常意味着 ONNX Runtime 将在需要更多内存时扩展内存池。
      cuda_options.do_copy_in_default_stream = 1;                              // 行设置在默认流中进行复制操作为 1，这意味着 ONNX Runtime 将在 CUDA 的默认流中进行数据复制操作。
      cuda_options.has_user_compute_stream   = 0;                              // 这行设置用户计算流为 0，这意味着 ONNX Runtime 将使用其自己的计算流，而不是用户提供的计算流。
      cuda_options.default_memory_arena_cfg  = nullptr;                        // 这行设置默认内存区配置为 nullptr，这意味着 ONNX Runtime 将使用默认的内存区配置。

      session_options_extractor_.AppendExecutionProvider_CUDA( cuda_options );
      session_options_extractor_.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_EXTENDED );
      session_options_matcher_.AppendExecutionProvider_CUDA( cuda_options );
      session_options_matcher_.SetGraphOptimizationLevel( GraphOptimizationLevel::ORT_ENABLE_EXTENDED );
    }

    INFO( logger, "Loading extractor model from {0} and matcher model from {1}", config.extractor_path, config.matcher_path );
    session_uptr_extractor_ = std::make_unique<Ort::Session>( env_extractor_, config.extractor_path.c_str(), session_options_extractor_ );
    session_uptr_matcher_   = std::make_unique<Ort::Session>( env_matcher_, config.matcher_path.c_str(), session_options_matcher_ );

    // get input node names and shapes
    // extractor input node names and shapes
    INFO( logger, "extractor input node names and shapes" );
    extractNodesInfo( IO{ INPUT }, input_node_names_extractor_, input_node_shapes_extractor_, session_uptr_extractor_, allocator_ );
    INFO( logger, "extractor output node names and shapes" );
    extractNodesInfo( IO{ OUTPUT }, output_node_names_extractor_, output_node_shapes_extractor_, session_uptr_extractor_, allocator_ );

    // matcher input node names and shapes
    INFO( logger, "matcher input node names and shapes" );
    extractNodesInfo( IO{ INPUT }, input_node_names_matcher_, input_node_shapes_matcher_, session_uptr_matcher_, allocator_ );
    INFO( logger, "matcher output node names and shapes" );
    extractNodesInfo( IO{ OUTPUT }, output_node_names_matcher_, output_node_shapes_matcher_, session_uptr_matcher_, allocator_ );

    INFO( logger, "ONNX Runtime environment initialized successfully!" );
  }
  catch ( const std::exception& e )
  {
    ERROR( logger, "ONNX Runtime environment initialized failed! Error message: {0}", e.what() );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

cv::Mat DecoupledOnnxRunner::preProcessExtractor( const Config& config, const cv::Mat& image, float& scale )
{
  float   scale_temp = scale;
  cv::Mat image_temp = image.clone();
  INFO( logger, "image_temp size: [{0}, {1}]", image_temp.cols, image_temp.rows );

  std::string fn{ "max" };
  std::string interp{ "area" };
  cv::Mat     image_result = normalizeImage( resizeImage( image_temp, config.image_size, scale, fn, interp ) );
  INFO( logger, "image_result size: [{0}, {1}], scale from {2} to {3}", image_result.cols, image_result.rows, scale_temp, scale );

  return image_result;
}

int DecoupledOnnxRunner::inferenceExtractor( const Config& config, const cv::Mat& image )
{
  INFO( logger, "**** extractor inference start ****" );

  try
  {
    INFO( logger, "image shape: [{0}, {1}], channel: {2}", image.cols, image.rows, image.channels() );

    // build input node shape
    // only support super point
    input_node_shapes_extractor_[ 0 ]    = { 1, 1, image.size().height, image.size().width };
    int                input_tensor_size = input_node_shapes_extractor_[ 0 ][ 0 ] * input_node_shapes_extractor_[ 0 ][ 1 ] * input_node_shapes_extractor_[ 0 ][ 2 ] * input_node_shapes_extractor_[ 0 ][ 3 ];
    std::vector<float> input_tensor_values_src( input_tensor_size );
    input_tensor_values_src.assign( image.begin<float>(), image.end<float>() );

    // create input tensor object from data values
    INFO( logger, "creating memory info handler" );
    auto memory_info_handler = Ort::MemoryInfo::CreateCpu( OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU );
    // auto memory_info_handler = Ort::MemoryInfo::CreateCpu( OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault );

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, input_tensor_values_src.data(), input_tensor_values_src.size(), input_node_shapes_extractor_[ 0 ].data(), input_node_shapes_extractor_[ 0 ].size() ) );

    // run
    timer_extractor_.tic();
    auto output_tensor_temp = session_uptr_extractor_->Run( Ort::RunOptions{ nullptr }, input_node_names_extractor_.data(), input_tensors.data(), input_tensors.size(), output_node_names_extractor_.data(), output_node_names_extractor_.size() );
    INFO( logger, "inference extractor time consumed: {0}", timer_extractor_.tocGetDuration() );

    int count = 0;
    for ( const auto& tensor : output_tensor_temp )
    {
      if ( !tensor.IsTensor() )
      {
        ERROR( logger, "Output {0} is not a tensor", count );
      }

      if ( !tensor.HasValue() )
      {
        ERROR( logger, "Output {0} is empty", count );
      }

      count++;
    }

    output_tensors_extoractor_.emplace_back( std::move( output_tensor_temp ) );
    INFO( logger, "inference success." );
  }
  catch ( const std::exception& e )
  {
    ERROR( logger, "**** inference failed with error message: {0} ****", e.what() );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

std::pair<std::vector<cv::Point2f>, float*> DecoupledOnnxRunner::postProcessExtractor( const Config& config, std::vector<Ort::Value> tensor )
{
  INFO( logger, "**** extractor post process start ****" );
  std::pair<std::vector<cv::Point2f>, float*> extractor_result;

  try
  {
    std::vector<int64_t> key_points_shape = tensor[ 0 ].GetTensorTypeAndShapeInfo().GetShape();
    int64_t*             key_points_ptr   = (int64_t*)tensor[ 0 ].GetTensorMutableData<void>();
    INFO( logger, "key points shape: [{0}, {1}, {2}]", key_points_shape[ 0 ], key_points_shape[ 1 ], key_points_shape[ 2 ] );

    std::vector<int64_t> score_shape = tensor[ 1 ].GetTensorTypeAndShapeInfo().GetShape();
    float*               scores_ptr  = (float*)tensor[ 1 ].GetTensorMutableData<void>();
    INFO( logger, "scores shape: [{0}, {1}, {2}]", score_shape[ 0 ], score_shape[ 1 ], score_shape[ 2 ] );


    std::vector<int64_t> descriptor_shape = tensor[ 2 ].GetTensorTypeAndShapeInfo().GetShape();
    float*               descriptors_ptr  = (float*)tensor[ 2 ].GetTensorMutableData<void>();
    INFO( logger, "descriptor shape: [{0}, {1}, {2}]", descriptor_shape[ 0 ], descriptor_shape[ 1 ], descriptor_shape[ 2 ] );

    // Process key points and descriptors
    std::vector<cv::Point2f> key_points;
    int                      count_poor   = 0;
    int                      count_strong = 0;
    for ( int i = 0; i < key_points_shape[ 1 ] * 2; i += 2 )
    {
      if ( scores_ptr[ i ] > 0.005f )
      {
        count_strong++;
      }
      else
      {
        count_poor++;
      }
      key_points.emplace_back( cv::Point2f( key_points_ptr[ i ], key_points_ptr[ i + 1 ] ) );
      // INFO( logger, "x y score --> [{0}, {1}, {2}]", key_points_ptr[ i ], key_points_ptr[ i + 1 ], scores_ptr[ i ] );
    }

    extractor_result.first  = key_points;
    extractor_result.second = descriptors_ptr;

    INFO( logger, "extractor post process success with {0} key points, {1} / {2}.", key_points.size(), count_poor, count_strong );
  }
  catch ( const std::exception& e )
  {
    ERROR( logger, "**** extractor post process failed with error message: {0} ****", e.what() );
    return std::make_pair( std::vector<cv::Point2f>{}, nullptr );
  }

  return extractor_result;
}

std::vector<cv::Point2f> DecoupledOnnxRunner::preProcessMatcher( std::vector<cv::Point2f> key_points, const int& height, const int& width )
{
  return normalizeKeyPoints( key_points, height, width );
}

int DecoupledOnnxRunner::inferenceMatcher( const Config& config, std::vector<cv::Point2f> key_points_src, std::vector<cv::Point2f> key_points_dst, float* descriptor_src, float* descriptor_dst )
{
  INFO( logger, "**** matcher inference start ****" );
  try
  {
    input_node_shapes_matcher_[ 0 ] = { 1, static_cast<int>( key_points_src.size() ), 2 };
    input_node_shapes_matcher_[ 1 ] = { 1, static_cast<int>( key_points_dst.size() ), 2 };
    input_node_shapes_matcher_[ 2 ] = { 1, static_cast<int>( key_points_src.size() ), 256 };
    input_node_shapes_matcher_[ 3 ] = { 1, static_cast<int>( key_points_dst.size() ), 256 };

    // create input tensor object from data values
    INFO( logger, "creating memory info handler" );
    auto memory_info_handler = Ort::MemoryInfo::CreateCpu( OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU );

    INFO( logger, "preparing input tensors" );
    float* key_points_src_data = new float[ key_points_src.size() * 2 ];
    float* key_points_dst_data = new float[ key_points_dst.size() * 2 ];

    INFO( logger, "trans to one dimension" );
    // trans to one dimension
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

    INFO( logger, "constructing input tensors" );
    std::vector<Ort::Value> input_tensors;
    // input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, key_points_src_data, key_points_src.size() * 2 * sizeof( float ), input_node_shapes_matcher_[ 0 ].data(), input_node_shapes_matcher_[ 0 ].size() ) );
    // input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, key_points_dst_data, key_points_dst.size() * 2 * sizeof( float ), input_node_shapes_matcher_[ 1 ].data(), input_node_shapes_matcher_[ 1 ].size() ) );
    // input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, descriptor_src, key_points_src.size() * 256 * sizeof( float ), input_node_shapes_matcher_[ 2 ].data(), input_node_shapes_matcher_[ 2 ].size() ) );
    // input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, descriptor_dst, key_points_dst.size() * 256 * sizeof( float ), input_node_shapes_matcher_[ 3 ].data(), input_node_shapes_matcher_[ 3 ].size() ) );

    input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, key_points_src_data, key_points_src.size() * 2, input_node_shapes_matcher_[ 0 ].data(), input_node_shapes_matcher_[ 0 ].size() ) );
    input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, key_points_dst_data, key_points_dst.size() * 2, input_node_shapes_matcher_[ 1 ].data(), input_node_shapes_matcher_[ 1 ].size() ) );
    input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, descriptor_src, key_points_src.size() * 256, input_node_shapes_matcher_[ 2 ].data(), input_node_shapes_matcher_[ 2 ].size() ) );
    input_tensors.push_back( Ort::Value::CreateTensor<float>( memory_info_handler, descriptor_dst, key_points_dst.size() * 256, input_node_shapes_matcher_[ 3 ].data(), input_node_shapes_matcher_[ 3 ].size() ) );


    for ( int i = 0; i < input_node_names_matcher_.size(); i++ )
    {
      INFO( logger, "input node {0} name: {1}", i, input_node_names_matcher_[ i ] );

      Ort::TypeInfo type_info   = session_uptr_matcher_->GetInputTypeInfo( i );
      auto          tensor_info = type_info.GetTensorTypeAndShapeInfo();
      auto          shape       = tensor_info.GetShape();
      INFO( logger, "input node shape size: {0}, shape: {1}, {2}, {3}", shape.size(), shape[ 0 ], shape[ 1 ], shape[ 2 ] );
      INFO( logger, "input node shape size: {0}, shape: {1}, {2}, {3}", input_node_shapes_matcher_[ i ].size(), input_node_shapes_matcher_[ i ][ 0 ], input_node_shapes_matcher_[ i ][ 1 ], input_node_shapes_matcher_[ i ][ 2 ] );
    }


    for ( const auto& name : output_node_names_matcher_ )
    {
      INFO( logger, "output node name: {0}", name );
    }

    timer_extractor_.tic();
    auto output_tensor_temp = session_uptr_matcher_->Run( Ort::RunOptions{ nullptr }, input_node_names_matcher_.data(), input_tensors.data(), input_tensors.size(), output_node_names_matcher_.data(), output_node_names_matcher_.size() );
    INFO( logger, "matcher inference time consumed: {0}", timer_extractor_.tocGetDuration() );

    int count = 0;
    for ( const auto& tensor : output_tensor_temp )
    {
      if ( !tensor.IsTensor() )
      {
        ERROR( logger, "Output {0} is not a tensor", count );
      }

      if ( !tensor.HasValue() )
      {
        ERROR( logger, "Output {0} is empty", count );
      }

      count++;
    }

    output_tensor_matcher_ = std::move( output_tensor_temp );
    INFO( logger, "matcher inference success." );
  }
  catch ( const std::exception& e )
  {
    ERROR( logger, "**** matcher inference failed with error message: {0} ****", e.what() );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int DecoupledOnnxRunner::postProcessMatcher( const Config& config, std::vector<cv::Point2f> key_points_src, std::vector<cv::Point2f> key_points_dst )
{
  INFO( logger, "**** matcher post process start ****" );

  try
  {
    std::vector<int64_t> matches_shape = output_tensor_matcher_[ 0 ].GetTensorTypeAndShapeInfo().GetShape();
    int64_t*             matches_ptr   = (int64_t*)output_tensor_matcher_[ 0 ].GetTensorMutableData<void>();
    for ( const auto& shape : matches_shape )
    {
      INFO( logger, "matches shape: {0}", shape );
    }


    std::vector<int64_t> scores_shape = output_tensor_matcher_[ 1 ].GetTensorTypeAndShapeInfo().GetShape();
    float*               scores_ptr   = (float*)output_tensor_matcher_[ 1 ].GetTensorMutableData<void>();
    for ( const auto& shape : scores_shape )
    {
      INFO( logger, "matches shape: {0}", shape );
    }

    // processing key points
    std::vector<cv::Point2f> key_points_src_temp;
    std::vector<cv::Point2f> key_points_dst_temp;
    for ( int i = 0; i < key_points_src.size(); ++i )
    {
      key_points_src_temp.emplace_back( cv::Point2f{ ( key_points_src[ i ].x + 0.5f ) / scales_[ 0 ] - 0.5f, ( key_points_src[ i ].y + 0.5f ) / scales_[ 0 ] - 0.5f } );
    }
    for ( int i = 0; i < key_points_dst.size(); ++i )
    {
      key_points_dst_temp.emplace_back( cv::Point2f{ ( key_points_dst[ i ].x + 0.5f ) / scales_[ 1 ] - 0.5f, ( key_points_dst[ i ].y + 0.5f ) / scales_[ 1 ] - 0.5f } );
    }

    // create matches indices
    std::set<std::pair<int, int>> matches;
    int                           count = 0;
    for ( int i = 0; i < matches_shape[ 0 ] * 2; i += 2 )
    {
      if ( matches_ptr[ i ] > -1 && matches_ptr[ count ] > match_threshold_ )
      {
        // INFO( logger, "matche pair index: [{0}, {1}] score: {2}", matches_0_ptr[ i ], matches_0_ptr[ i + 1 ], match_scores_0_ptr[ count ] );
        matches.insert( std::make_pair( matches_ptr[ i ], matches_ptr[ i + 1 ] ) );
      }
      count++;
    }


    std::vector<cv::Point2f> key_points_src, key_points_dst;
    for ( const auto& match : matches )
    {
      key_points_src.emplace_back( key_points_src_temp[ match.first ] );
      key_points_dst.emplace_back( key_points_dst_temp[ match.second ] );
    }

    key_points_result_.first  = key_points_src;
    key_points_result_.second = key_points_dst;
    key_points_src_           = key_points_src_temp;
    key_points_dst_           = key_points_dst_temp;

    INFO( logger, "matches size: {0}", matches.size() );
  }
  catch ( const std::exception& e )
  {
    ERROR( logger, "**** post process failed with error message: {0} ****", e.what() );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> DecoupledOnnxRunner::inferenceImagePair( const Config& config, const cv::Mat& image_src, const cv::Mat& image_dst )
{
  INFO( logger, "**** inference image pair start ****" );

  if ( image_src.empty() || image_dst.empty() )
  {
    ERROR( logger, "image src or image dst is empty" );
    return std::make_pair( std::vector<cv::Point2f>{}, std::vector<cv::Point2f>{} );
  }

  cv::Mat image_src_copy = image_src.clone();
  cv::Mat image_dst_copy = image_dst.clone();

  INFO( logger, "preprocessing image_src" );
  cv::Mat image_src_temp = preProcessExtractor( config, image_src_copy, scales_[ 0 ] );
  INFO( logger, "preprocessing image_dst" );
  cv::Mat image_dst_temp = preProcessExtractor( config, image_dst_copy, scales_[ 1 ] );

  inferenceExtractor( config, image_src_temp );
  inferenceExtractor( config, image_dst_temp );

  auto extractor_result_src = postProcessExtractor( config, std::move( output_tensors_extoractor_[ 0 ] ) );
  auto extractor_result_dst = postProcessExtractor( config, std::move( output_tensors_extoractor_[ 1 ] ) );

  // matcher
  auto key_points_src_normalized = preProcessMatcher( extractor_result_src.first, image_src_temp.rows, image_src_temp.cols );
  auto key_points_dst_normalized = preProcessMatcher( extractor_result_dst.first, image_dst_temp.rows, image_dst_temp.cols );

  inferenceMatcher( config, key_points_src_normalized, key_points_dst_normalized, extractor_result_src.second, extractor_result_dst.second );
  postProcessMatcher( config, extractor_result_src.first, extractor_result_dst.first );

  output_tensor_matcher_.clear();
  output_tensors_extoractor_.clear();

  return getKeyPointsResult();
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> DecoupledOnnxRunner::getKeyPointsResult() const
{
  return key_points_result_;
}

std::vector<cv::Point2f> DecoupledOnnxRunner::getKeyPointsSrc() const
{
  return key_points_src_;
}

std::vector<cv::Point2f> DecoupledOnnxRunner::getKeyPointsDst() const
{
  return key_points_dst_;
}

float DecoupledOnnxRunner::getMatchThreshold() const
{
  return match_threshold_;
}

void DecoupledOnnxRunner::setMatchThreshold( float match_threshold )
{
  match_threshold_ = match_threshold;
}