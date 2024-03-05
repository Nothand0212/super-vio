#include <onnxruntime_cxx_api.h>

#include <iostream>

int main()
{
  // 创建一个 ONNX Runtime 环境
  Ort::Env env( ORT_LOGGING_LEVEL_WARNING, "Test" );

  // 检查 ONNX Runtime 是否能正常运行
  if ( env )
  {
    std::cout << "ONNX Runtime is working correctly." << std::endl;
  }
  else
  {
    std::cout << "Failed to initialize ONNX Runtime." << std::endl;
  }

  return 0;
}