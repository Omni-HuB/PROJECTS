#include "sim/sim_object.hh"
#include "vector_operations.hh"
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

void VectorOperations::init()
{
    Eigen::Vector3d vector1(1.0, 2.0, 3.0); // Initial vector 1
    Eigen::Vector3d vector2(4.0, 5.0, 6.0); // Initial vector 2

    if (debug::Vector) {
        std::cout << "Initial Vectors:\n";
        std::cout << "Vector 1: " << vector1 << "\n";
        std::cout << "Vector 2: " << vector2 << "\n";
    }

    // Perform VectorCrossProduct at tick 150
    schedule(event, 150);

    // Perform NormalizeVector at tick 1500
    schedule(event, 1500);

    // Perform VectorSubtraction at tick 15000
    schedule(event, 15000);
}

void VectorOperations::event()
{
    Eigen::Vector3d vector1(1.0, 2.0, 3.0); // Initial vector 1
    Eigen::Vector3d vector2(4.0, 5.0, 6.0); // Initial vector 2

    if (curTick() == 150) {
        // Calculate and print the VectorCrossProduct
        if (debug::ResultCross) {
            std::cout << "VectorCrossProduct Result:\n";

            // Calculate the cross product
            Eigen::Vector3d crossProduct = vector1.cross(vector2);

            // Print the result
            std::cout << "Cross Product: " << crossProduct << "\n";
        }
    }
    else if (curTick() == 1500) {
        // Calculate and print normalized vectors
        if (debug::Normalize) {
            std::cout << "Normalized Vectors:\n";

            // Calculate the normalized vectors
            Eigen::Vector3d normalizedVector1 = vector1.normalized();
            Eigen::Vector3d normalizedVector2 = vector2.normalized();

            // Print the normalized vectors
            std::cout << "Normalized Vector 1: " << normalizedVector1 << "\n";
            std::cout << "Normalized Vector 2: " << normalizedVector2 << "\n";
        }
    }
    else if (curTick() == 15000) {
        // Calculate and print the result of VectorSubtraction
        if (debug::ResultSub) {
            std::cout << "VectorSubtraction Result:\n";

            // Calculate the vector subtraction
            Eigen::Vector3d result = vector1 - vector2;

            // Print the result
            std::cout << "Result of Vector Subtraction: " << result << "\n";
        }
    }
}

// Define DEBUG flags
DebugFlag(Vector, "Enable debugging for vectors");
DebugFlag(ResultCross, "Enable debugging for VectorCrossProduct result");
DebugFlag(Normalize, "Enable debugging for NormalizeVector result");
DebugFlag(ResultSub, "Enable debugging for VectorSubtraction result");

// Instantiate the SimObject
VectorOperations vector_operations = {
    "VectorOperations",
    "VectorOperations",
    NULL,
    &vector_operations,
};
