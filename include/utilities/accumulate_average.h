#pragma once

#include <cmath>

#include "logger/logger.h"
#include "utilities/color.h"
class AccumulateAverage
{
public:
  AccumulateAverage() : count_( 0 ), average_( 0 ) {}

  void addValue( double value )
  {
    if ( !isnanf( value ) )
    {
      last_value_ = value;
      average_    = ( average_ * count_ + value ) / ( count_ + 1 );
      count_++;
    }
    else
    {
      WARN( logger, "**** Adding NaN Value! ****" );
    }
  }

  double getAverage() const
  {
    return average_;
  }

  double getLastValue() const
  {
    return last_value_;
  }

private:
  int    count_;
  double average_;
  double last_value_;
};