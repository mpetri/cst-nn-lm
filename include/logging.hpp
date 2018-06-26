#pragma once

#include <locale>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>

#define BOOST_LOG_DYN_LINK 1


namespace logging = boost::log;
namespace src = boost::log::sources;
namespace sinks = boost::log::sinks;
namespace keywords = boost::log::keywords;

#define CNLOG BOOST_LOG_TRIVIAL(info)

void init_logging()
{
    boost::log::core::get()->remove_all_sinks();
    boost::log::core::get()->add_global_attribute("TimeStamp", boost::log::attributes::local_clock());
    logging::add_console_log(std::cout,keywords::format = "[%TimeStamp%]: %Message%");
    std::cout.imbue( std::locale( "C.UTF-8" ) ) ; 
}

