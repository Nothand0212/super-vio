#include "extractor/extractor.h"
#include "logger/logger.h"

Extracotr::Extracotr( unsigned int threads_num, unsigned int point_num )
    : m_threads_num{ threads_num }, m_point_num{ point_num }
{
  INFO( logger, "Extractor is being constructed, threads_num: {0}, point_num: {1}", threads_num, point_num );
}

Extracotr::~Extracotr()
{
  INFO( logger, "Extractor is being destructed" );
}

int Extracotr::initOrtEnv( const Config& config )
{
  INFO( logger, "Initializing extractor ort env" );

  try
  {
    m_env             = Ort::Env( ORT_LOGGING_LEVEL_WARNING, "Extractor" );
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
      INFO( logger, "Using CUDA for extractor" );

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

    INFO( logger, "Loading extractor model from {0}", config.extractor_path );

    m_uptr_session = std::make_unique<Ort::Session>( m_env, config.extractor_path.c_str(), m_session_options );

    INFO( logger, "Extractor model loaded" );
    extractNodesInfo( IO::INPUT, m_vec_input_names, m_vec_input_shapes, m_uptr_session, m_allocator );
    extractNodesInfo( IO::OUTPUT, m_vec_output_names, m_vec_output_shapes, m_uptr_session, m_allocator );

    INFO( logger, "Extractor ort env initialized" );
  }
  catch ( const std::exception& e )
  {
    INFO( logger, "Failed to initialize extractor ort env: {0}", e.what() );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

cv::Mat Extracotr::prePorcess( const Config& config, const cv::Mat& image, float& scale )
{
  INFO( logger, "Preprocessing image" );

  float   scale_temp = scale;
  cv::Mat image_temp = image.clone();
  INFO( logger, "image_in size: [{0}, {1}]", image_temp.cols, image_temp.rows );

  std::string fn{ "max" };
  std::string interp{ "area" };
  cv::Mat     image_result = normalizeImage( resizeImage( image_temp, config.image_size, scale, fn, interp ) );
  INFO( logger, "image_out size: [{0}, {1}], scale from 1.0 to {3}", image_result.cols, image_result.rows, scale_temp, scale );

  m_width_transformed  = image_result.cols;
  m_height_transformed = image_result.rows;
  m_scale              = scale;
  return image_result;
}

int Extracotr::inference( const Config& config, const cv::Mat& image )
{
  INFO( logger, "Inferencing image" );

  try
  {
    m_vec_input_shapes[ 0 ] = { 1, 1, image.size().height, image.size().width };
    int input_tensor_size   = m_vec_input_shapes[ 0 ][ 0 ] * m_vec_input_shapes[ 0 ][ 1 ] * m_vec_input_shapes[ 0 ][ 2 ] * m_vec_input_shapes[ 0 ][ 3 ];

    std::vector<float> input_tensor_values( input_tensor_size );
    input_tensor_values.assign( image.begin<float>(), image.end<float>() );

    auto memory_info_handler = Ort::MemoryInfo::CreateCpu( OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU );

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back( Ort::Value::CreateTensor<float>( memory_info_handler, input_tensor_values.data(), input_tensor_size, m_vec_input_shapes[ 0 ].data(), m_vec_input_shapes[ 0 ].size() ) );

    m_timer.tic();
    auto output_tensors = m_uptr_session->Run( Ort::RunOptions{ nullptr }, m_vec_input_names.data(), input_tensors.data(), input_tensors.size(), m_vec_output_names.data(), m_vec_output_names.size() );
    INFO( logger, "inference extractor time consumed: {0}", m_timer.tocGetDuration() );


    int count = 0;
    for ( const auto& tensor : output_tensors )
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

    m_key_points = postProcess( config, std::move( output_tensors ) );
  }
  catch ( const std::exception& e )
  {
    ERROR( logger, "Failed to inference image: {0}", e.what() );
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

Features Extracotr::postProcess( const Config& config, std::vector<Ort::Value> tensor )
{
  INFO( logger, "Transforming tensor to key points" );
  Features key_points_result;

  try
  {
    std::vector<int64_t> points_shape     = tensor[ 0 ].GetTensorTypeAndShapeInfo().GetShape();
    int64_t*             ptr_points_value = (int64_t*)tensor[ 0 ].GetTensorMutableData<int64_t>();
    INFO( logger, "key points shape: [{0}, {1}, {2}], size: {3}", points_shape[ 0 ], points_shape[ 1 ], points_shape[ 2 ], points_shape.size() );

    std::vector<cv::Point2f> vec_points;
    for ( int i = 0; i < points_shape[ 1 ] * 2; i += 2 )
    {
      vec_points.emplace_back( cv::Point2f( ( ptr_points_value[ i ] + 0.5f ) / m_scale - 0.5f, ( ptr_points_value[ i + 1 ] + 0.5f ) / m_scale - 0.5f ) );
    }
    key_points_result.setKeyPoints( vec_points );

    std::vector<int64_t> score_shape     = tensor[ 1 ].GetTensorTypeAndShapeInfo().GetShape();
    float*               ptr_score_value = (float*)tensor[ 1 ].GetTensorMutableData<float>();
    INFO( logger, "score shape: [{0}, {1}], size: {2}", score_shape[ 0 ], score_shape[ 1 ], score_shape.size() );

    std::vector<float> vec_score;
    for ( int i = 0; i < score_shape[ 1 ]; i++ )
    {
      vec_score.emplace_back( ptr_score_value[ i ] );
    }
    key_points_result.setScores( vec_score );

    std::vector<int64_t> descriptor_shape     = tensor[ 2 ].GetTensorTypeAndShapeInfo().GetShape();
    float*               ptr_descriptor_value = (float*)tensor[ 2 ].GetTensorMutableData<float>();
    INFO( logger, "descriptor shape: [{0}, {1}, {2}], size: {3}", descriptor_shape[ 0 ], descriptor_shape[ 1 ], descriptor_shape[ 2 ], descriptor_shape.size() );

    cv::Mat mat_descriptor( descriptor_shape[ 1 ], descriptor_shape[ 2 ], CV_32FC1, ptr_descriptor_value );
    INFO( logger, "descriptor size: [{0}, {1}]", mat_descriptor.cols, mat_descriptor.rows );
    key_points_result.setDescriptor( mat_descriptor );
  }
  catch ( const std::exception& e )
  {
    ERROR( logger, "Failed to transform tensor to key points: {0}", e.what() );
    return Features{};
  }

  return key_points_result;
}


std::pair<Features, std::vector<Region>> Extracotr::distributeKeyPointsDebug( const Features& key_points, const cv::Mat& image )
{
  INFO( logger, "Distributing key points" );

  //   if ( m_point_num == 0 || key_points.getScores().size() <= m_point_num )
  //   {
  //     WARN( logger, "Point number is {0}, m_point_num_is {1}, no need to distribute", key_points.getScores().size(), m_point_num );
  //     return std::pair<Features, std::vector<Region>>{};
  //   }

  cv::Rect2i      rect( 0, 0, image.cols, image.rows );
  std::deque<int> indexs( static_cast<int>( key_points.getScores().size() ) );
  for ( int i = 0; i < indexs.size(); i++ )
  {
    indexs[ i ] = i;
  }
  Region region{ rect, indexs };

  std::deque<Region> queue_region;
  queue_region.push_back( region );

  int count               = 0;
  int last_region_size    = 0;
  int current_region_size = queue_region.size();

  while ( queue_region.size() < m_point_num && !queue_region.empty() )
  {
    INFO( logger, "Regions number: {0} / {1}", queue_region.size(), last_region_size );

    if ( current_region_size == last_region_size )
    {
      count++;
      if ( count > 10 )
      {
        break;
      }
    }
    else
    {
      last_region_size = current_region_size;
    }

    Region temp_region = queue_region.front();
    queue_region.pop_front();

    std::deque<int> temp_indexs = temp_region.indexs;
    // INFO( logger, "temp_indexs size: {0}", temp_indexs.size() );

    cv::Rect2i temp_rect = temp_region.rectangle;
    // INFO( logger, "temp_rect: [{0}, {1}, {2}, {3}]", temp_rect.x, temp_rect.y, temp_rect.width, temp_rect.height );

    int mid_x = temp_rect.x + temp_rect.width / 2;
    int mid_y = temp_rect.y + temp_rect.height / 2;

    cv::Rect2i      top_left_rect( temp_rect.x, temp_rect.y, mid_x - temp_rect.x, mid_y - temp_rect.y );
    std::deque<int> top_left_indexs;

    cv::Rect2i      top_right_rect( mid_x, temp_rect.y, temp_rect.x + temp_rect.width - mid_x, top_left_rect.height );
    std::deque<int> top_right_indexs;

    cv::Rect2i      bottom_left_rect( temp_rect.x, mid_y, mid_x - temp_rect.x, temp_rect.y + temp_rect.height - mid_y );
    std::deque<int> bottom_left_indexs;

    cv::Rect2i      bottom_right_rect( mid_x, mid_y, temp_rect.width - mid_x, temp_rect.height - mid_y );
    std::deque<int> bottom_right_indexs;

    while ( !temp_indexs.empty() )
    {
      int index = temp_indexs.front();
      temp_indexs.pop_front();

      cv::Point2f point = key_points.getKeyPoints()[ index ];
      if ( point.x < mid_x )
      {
        if ( point.y < mid_y )
        {
          top_left_indexs.push_back( index );
        }
        else
        {
          bottom_left_indexs.push_back( index );
        }
      }
      else
      {
        if ( point.y < mid_y )
        {
          top_right_indexs.push_back( index );
        }
        else
        {
          bottom_right_indexs.push_back( index );
        }
      }
    }


    if ( !top_left_indexs.empty() )
    {
      queue_region.push_back( Region{ top_left_rect, top_left_indexs } );
    }
    else
    {
      ERROR( logger, "top_left_indexs is empty" );
    }


    if ( !top_right_indexs.empty() )
    {
      queue_region.push_back( Region{ top_right_rect, top_right_indexs } );
    }
    else
    {
      ERROR( logger, "top_right_indexs is empty" );
    }

    if ( !bottom_left_indexs.empty() )
    {
      queue_region.push_back( Region{ bottom_left_rect, bottom_left_indexs } );
    }
    else
    {
      ERROR( logger, "bottom_left_indexs is empty" );
    }


    if ( !bottom_right_indexs.empty() )
    {
      queue_region.push_back( Region{ bottom_right_rect, bottom_right_indexs } );
    }
    else
    {
      ERROR( logger, "bottom_right_indexs is empty" );
    }

    current_region_size = queue_region.size();
  }

  std::vector<Region>      vec_regions;
  std::vector<cv::Point2f> vec_points;
  std::vector<float>       vec_score;
  cv::Mat                  mat_descriptor;
  std::vector<int>         best_indexs;
  while ( !queue_region.empty() )
  {
    Region temp_region = queue_region.front();
    vec_regions.push_back( temp_region );
    queue_region.pop_front();

    std::deque<int> temp_indexs = temp_region.indexs;
    float           MAX_SCORE{ -1.0f };
    int             best_index = -1;
    while ( !temp_indexs.empty() )
    {
      int index = temp_indexs.front();
      temp_indexs.pop_front();

      if ( key_points.getScores()[ index ] > MAX_SCORE )
      {
        MAX_SCORE  = key_points.getScores()[ index ];
        best_index = index;
      }
    }
    if ( best_index != -1 )
    {
      best_indexs.push_back( best_index );
    }
    else
    {
      ERROR( logger, "best_index is -1" );
    }
  }

  for ( const auto& index : best_indexs )
  {
    vec_points.emplace_back( key_points.getKeyPoints()[ index ] );
    vec_score.emplace_back( key_points.getScores()[ index ] );
    mat_descriptor.push_back( key_points.getDescriptor().row( index ) );
  }

  Features key_points_result{};
  key_points_result.setKeyPoints( vec_points );
  key_points_result.setScores( vec_score );
  key_points_result.setDescriptor( mat_descriptor );

  return std::pair<Features, std::vector<Region>>{ key_points_result, vec_regions };
}

Features Extracotr::distributeKeyPoints( const Features& key_points, const cv::Mat& image )
{
  INFO( logger, "Distributing key points" );

  if ( m_point_num == 0 || key_points.getScores().size() <= m_point_num )
  {
    WARN( logger, "Point number is {0}, m_point_num_is {1}, no need to distribute", key_points.getScores().size(), m_point_num );
    return key_points;
  }

  cv::Rect2i      rect( 0, 0, image.cols, image.rows );
  std::deque<int> indexs( static_cast<int>( key_points.getScores().size() ) );
  for ( int i = 0; i < indexs.size(); i++ )
  {
    indexs[ i ] = i;
  }
  Region region{ rect, indexs };

  std::deque<Region> queue_region;
  queue_region.push_back( region );

  while ( queue_region.size() < m_point_num && !queue_region.empty() )
  {
    INFO( logger, "Regions number: {0}", queue_region.size() );
    Region temp_region = queue_region.front();
    queue_region.pop_front();

    std::deque<int> temp_indexs = temp_region.indexs;
    INFO( logger, "temp_indexs size: {0}", temp_indexs.size() );

    cv::Rect2i temp_rect = temp_region.rectangle;
    INFO( logger, "temp_rect: [{0}, {1}, {2}, {3}]", temp_rect.x, temp_rect.y, temp_rect.width, temp_rect.height );

    int mid_x = temp_rect.x + temp_rect.width / 2;
    int mid_y = temp_rect.y + temp_rect.height / 2;

    cv::Rect2i      top_left_rect( temp_rect.x, temp_rect.y, mid_x - temp_rect.x, mid_y - temp_rect.y );
    std::deque<int> top_left_indexs;

    cv::Rect2i      top_right_rect( mid_x, temp_rect.y, temp_rect.x + temp_rect.width - mid_x, top_left_rect.height );
    std::deque<int> top_right_indexs;

    cv::Rect2i      bottom_left_rect( temp_rect.x, mid_y, mid_x - temp_rect.x, temp_rect.y + temp_rect.height - mid_y );
    std::deque<int> bottom_left_indexs;

    cv::Rect2i      bottom_right_rect( mid_x, mid_y, temp_rect.width - mid_x, temp_rect.height - mid_y );
    std::deque<int> bottom_right_indexs;

    while ( !temp_indexs.empty() )
    {
      int index = temp_indexs.front();
      temp_indexs.pop_front();

      cv::Point2f point = key_points.getKeyPoints()[ index ];
      if ( point.x < mid_x )
      {
        if ( point.y < mid_y )
        {
          top_left_indexs.push_back( index );
        }
        else
        {
          bottom_left_indexs.push_back( index );
        }
      }
      else
      {
        if ( point.y < mid_y )
        {
          top_right_indexs.push_back( index );
        }
        else
        {
          bottom_right_indexs.push_back( index );
        }
      }
    }


    if ( !top_left_indexs.empty() )
    {
      queue_region.push_back( Region{ top_left_rect, top_left_indexs } );
    }
    else
    {
      ERROR( logger, "top_left_indexs is empty" );
    }


    if ( !top_right_indexs.empty() )
    {
      queue_region.push_back( Region{ top_right_rect, top_right_indexs } );
    }
    else
    {
      ERROR( logger, "top_right_indexs is empty" );
    }

    if ( !bottom_left_indexs.empty() )
    {
      queue_region.push_back( Region{ bottom_left_rect, bottom_left_indexs } );
    }
    else
    {
      ERROR( logger, "bottom_left_indexs is empty" );
    }


    if ( !bottom_right_indexs.empty() )
    {
      queue_region.push_back( Region{ bottom_right_rect, bottom_right_indexs } );
    }
    else
    {
      ERROR( logger, "bottom_right_indexs is empty" );
    }
  }


  std::vector<cv::Point2f> vec_points;
  std::vector<float>       vec_score;
  cv::Mat                  mat_descriptor;
  std::vector<int>         best_indexs;
  while ( !queue_region.empty() )
  {
    Region temp_region = queue_region.front();
    queue_region.pop_front();

    std::deque<int> temp_indexs = temp_region.indexs;
    float           MAX_SCORE{ -1.0f };
    int             best_index = -1;
    while ( !temp_indexs.empty() )
    {
      int index = temp_indexs.front();
      temp_indexs.pop_front();

      if ( key_points.getScores()[ index ] > MAX_SCORE )
      {
        MAX_SCORE  = key_points.getScores()[ index ];
        best_index = index;
      }
    }
    if ( best_index != -1 )
    {
      best_indexs.push_back( best_index );
    }
    else
    {
      ERROR( logger, "best_index is -1" );
    }
  }

  for ( const auto& index : best_indexs )
  {
    vec_points.emplace_back( key_points.getKeyPoints()[ index ] );
    vec_score.emplace_back( key_points.getScores()[ index ] );
    mat_descriptor.push_back( key_points.getDescriptor().row( index ) );
  }

  Features key_points_result{};
  key_points_result.setKeyPoints( vec_points );
  key_points_result.setScores( vec_score );
  key_points_result.setDescriptor( mat_descriptor );

  return key_points_result;
}


Features Extracotr::inferenceImage( const Config& config, const cv::Mat& image )
{
  INFO( logger, "**** Inferencing image ****" );

#ifdef DEBUG
  Timer timer;
  timer.tic();
#endif

  cv::Mat image_temp = prePorcess( config, image, m_scale );

  if ( inference( config, image_temp ) == EXIT_SUCCESS )
  {
#ifdef DEBUG
    INFO( logger, "inference image time consumed: {0}", timer.tocGetDuration() );
#endif
    return getKeyPoints();
  }
  else
  {
    ERROR( logger, "Failed to inference image" );
    return Features{};
  }
}

Features Extracotr::getKeyPoints() const
{
  return m_key_points;
}

float Extracotr::getScale() const
{
  return m_scale;
}

int Extracotr::getWidthTransformed() const
{
  return m_width_transformed;
}

int Extracotr::getHeightTransformed() const
{
  return m_height_transformed;
}
