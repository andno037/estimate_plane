#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <Eigen/QR>  // Include QR for least squares solving


// Function to estimate a plane from a set of points
void estimatePlane(const std::vector<Eigen::Vector3d>& points, Eigen::Vector3d& plane_normal, double& plane_d) {
    if (points.size() < 3) {
        std::cerr << "At least three points are required to estimate a plane." << std::endl;
        return;
    }

    // Compute the centroid of the points
    Eigen::Vector3d centroid(0, 0, 0);
    for (const auto& point : points) {
        centroid += point;
    }
    centroid /= points.size();

    // Construct the covariance matrix
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const auto& point : points) {
        Eigen::Vector3d centered_point = point - centroid;
        covariance += centered_point * centered_point.transpose();
    }

    // Perform Eigen decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance);
    if (eigen_solver.info() != Eigen::Success) {
        std::cerr << "Eigen decomposition failed!" << std::endl;
        return;
    }

    // The normal to the plane is the Eigenvector corresponding to the smallest Eigenvalue
    plane_normal = eigen_solver.eigenvectors().col(0);

    // Calculate plane_d using the plane equation: ax + by + cz + d = 0
    plane_d = -plane_normal.dot(centroid);
}

// Function to calculate the angle between the plane normal and the z-axis
double calculateAngleWithZAxis(const Eigen::Vector3d& plane_normal) {
    // Reference direction (z-axis)
    Eigen::Vector3d z_axis(0, 0, 1);

    // Dot product and magnitudes of vectors
    double dot_product = plane_normal.dot(z_axis);
    double magnitude_normal = plane_normal.norm();
    double magnitude_z = z_axis.norm();

    // Calculate the cosine of the angle
    double cos_angle = dot_product / (magnitude_normal * magnitude_z);

    // Ensure the value is within the valid range for arccos
    if (cos_angle > 1.0) cos_angle = 1.0;
    if (cos_angle < -1.0) cos_angle = -1.0;

    // Calculate and return the angle in degrees
    double angle_radians = std::acos(cos_angle);
    double angle_degrees = angle_radians * 180.0 / M_PI;

    return angle_degrees;
}
// Function to estimate a plane from a set of points using least squares
void estimatePlaneLeastSquares(const std::vector<Eigen::Vector3d>& points, Eigen::Vector3d& plane_normal, double& plane_d) {
    if (points.size() < 3) {
        std::cerr << "At least three points are required to estimate a plane." << std::endl;
        return;
    }

    // Construct the matrix A and vector b
    Eigen::MatrixXd A(points.size(), 3);
    Eigen::VectorXd b(points.size());

    for (size_t i = 0; i < points.size(); ++i) {
        A(i, 0) = points[i].x();
        A(i, 1) = points[i].y();
        A(i, 2) = 1.0;
        b(i) = points[i].z();
    }

    // Solve the linear system Ax = b using least squares
    Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);

    // The plane equation is ax + by + c = z, where [a, b, -1] is the normal and c is the offset
    plane_normal = Eigen::Vector3d(x(0), x(1), -1.0);
    plane_d = x(2);

    // Normalize the plane normal
    double norm = -plane_normal.norm();
    plane_normal /= norm;
    plane_d /= norm;
}

// Function to calculate the angle between the plane normal and the z-axis
double calculateAngleWithZAxis2(const Eigen::Vector3d& plane_normal) {
    // Reference direction (z-axis)
    Eigen::Vector3d z_axis(0, 0, 1);

    // Dot product and magnitudes of vectors
    double dot_product = plane_normal.dot(z_axis);
    double magnitude_normal = plane_normal.norm();
    double magnitude_z = z_axis.norm();

    // Calculate the cosine of the angle
    double cos_angle = dot_product / (magnitude_normal * magnitude_z);

    // Ensure the value is within the valid range for arccos
    if (cos_angle > 1.0) cos_angle = 1.0;
    if (cos_angle < -1.0) cos_angle = -1.0;

    // Calculate and return the angle in degrees
    double angle_radians = std::acos(cos_angle);
    double angle_degrees = angle_radians * 180.0 / M_PI;

    return angle_degrees;
}

int main() {
    // Example points (you can replace these with your own points)
    std::vector<Eigen::Vector3d> points = {
       // {1.0, 2.0, 3.0},
       // {4.0, 5.0, 6.0},
       // {7.0, 8.0, 9.0},
       // {10.0, 11.0, 12.0}
       {0.0, 0.0, 0.0},
       {0.0, 1.0, 0.0},
       {1.0, 0.0, 1.0},
       {1.0, 1.0, 1.0},
    };

    Eigen::Vector3d plane_normal;
    double plane_d;

    estimatePlane(points, plane_normal, plane_d);

    std::cout << "Estimated plane normal: " << plane_normal.transpose() << std::endl;
    std::cout << "Estimated plane d: " << plane_d << std::endl;

    double angle_with_z = calculateAngleWithZAxis(plane_normal);
    std::cout << "Angle with z-axis: " << angle_with_z << " degrees" << std::endl;


    estimatePlaneLeastSquares(points, plane_normal, plane_d);

    std::cout << "Estimated plane normal: " << plane_normal.transpose() << std::endl;
    std::cout << "Estimated plane d: " << plane_d << std::endl;
    double angle_with_z_2 = calculateAngleWithZAxis2(plane_normal);
    std::cout << "Angle with z-axis: " << angle_with_z_2 << " degrees" << std::endl;
    return 0;
}

