#ifndef __VECTOR_OPERATIONS_HH__
#define __VECTOR_OPERATIONS_HH__

#include "base/flags.hh"
#include "sim/sim_object.hh"

class VectorOperations : public SimObject
{
 public:
   VectorOperations() {}
   void init() override;
   void event();
};

#endif
