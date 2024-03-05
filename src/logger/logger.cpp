#include "logger/logger.h"

std::shared_ptr<spdlog::logger> logger;

void InitLogger( const std::string& log_path )
{
  auto console_logger_sptr = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto file_logger_sptr    = std::make_shared<spdlog::sinks::basic_file_sink_mt>( log_path, true );
  logger                   = std::make_shared<spdlog::logger>( "MineLog", spdlog::sinks_init_list{ console_logger_sptr, file_logger_sptr } );

  // Set the log format
  logger->set_pattern( "[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] [%@:%#] %v" );
}
