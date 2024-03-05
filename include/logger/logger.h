/**
 ******************************************************************************
 * @file           : include/combined_onnx_runner.cpp
 * @author         : lin
 * @email          : linzeshi@foxmail.com
 * @brief          : None
 * @attention      : None
 * @date           : 24-1-18
 ******************************************************************************
 */

#pragma once

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#define INFO  SPDLOG_LOGGER_INFO
#define WARN  SPDLOG_LOGGER_WARN
#define ERROR SPDLOG_LOGGER_ERROR
#define DEBUG SPDLOG_LOGGER_DEBUG
#define TRACE SPDLOG_LOGGER_TRACE

// 创建一个全局的spdlog对象
extern std::shared_ptr<spdlog::logger> logger;

void InitLogger( const std::string& log_path );
