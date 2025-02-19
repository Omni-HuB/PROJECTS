// VectorOperations.cc

#include "sim/sim_object.hh"
#include "base/debug.hh"
#include <iostream>
#include <Eigen/Dense>

namespace gem5
{

class VectorOperations : public SimObject
{
    typedef VectorOperations Self;
    typedef SimObject Parent;

  public:
    VectorOperations() {}
    void init() override;
    void VectorCrossProduct();
    void NormalizeVector();
    void VectorSubtraction();

  private:
    Eigen::Vector3d vector1;
    Eigen::Vector3d vector2;
};


void VectorOperations::init()
{
    DPRINTF(VECTOR, "VectorOperations initialized\n");
    // Hardcode the initial vectors as per the assignment requirements.
    vector1 << 1, 2, 3;
    vector2 << 4, 5, 6;

    // Schedule the events as per the assignment requirements.
    schedule(EventFunctionWrapper([this]{ VectorCrossProduct(); }, name()), 150);
    schedule(EventFunctionWrapper([this]{ NormalizeVector(); }, name()), 1500);
    schedule(EventFunctionWrapper([this]{ VectorSubtraction(); }, name()), 15000);
}

void VectorOperations::VectorCrossProduct()
{
    // Compute the cross product of vector1 and vector2
    Eigen::Vector3d crossProduct = vector1.cross(vector2);
    DPRINTF(RESULTCROSS, "Cross product result: <%f, %f, %f>\n", crossProduct.x(), crossProduct.y(), crossProduct.z());
}

void VectorOperations::NormalizeVector()
{
    // Normalize vector1 and vector2
    vector1.normalize();
    vector2.normalize();
    DPRINTF(NORMALIZE, "Normalized vectors: <%f, %f, %f> and <%f, %f, %f>\n",
            vector1.x(), vector1.y(), vector1.z(),
            vector2.x(), vector2.y(), vector2.z());
}

void VectorOperations::VectorSubtraction()
{
    // Compute the subtraction of vector1 from vector2
    Eigen::Vector3d subtractionResult = vector2 - vector1;
    DPRINTF(RESULTSUB, "Vector subtraction result: <%f, %f, %f>\n", subtractionResult.x(), subtractionResult.y(), subtractionResult.z());
}

VectorOperations*
VectorOperationsParams::create()
{
    return new VectorOperations();
}

// Add debug flags with descriptions/annotations
namespace VectorOperationsFlag {
    DebugVector VECTOR("VECTOR", "Display vectors");
    DebugVector RESULTCROSS("RESULTCROSS", "Display cross product result");
    DebugVector NORMALIZE("NORMALIZE", "Display normalized vectors");
    DebugVector RESULTSUB("RESULTSUB", "Display vector subtraction result");
}

}