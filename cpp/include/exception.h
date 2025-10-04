#pragma once

#include <stdexcept>
#include <string>

namespace axe {

/**
 * @class AxeException
 * @brief A custom exception class for the Axe library.
 *
 * This exception type is used throughout the C++ core to report errors
 * in a way that can be easily caught and translated into Python exceptions
 * by the pybind11 layer. It ensures that all errors originating from the
 * C++ backend are clearly identifiable.
 */
class AxeException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/**
 * @class OOMError
 * @brief An exception specific to out-of-memory errors.
 *
 * This allows for more specific error handling when memory allocation fails,
 * enabling features like the Smart OOM Handler.
 */
class OOMError : public AxeException {
public:
    using AxeException::AxeException;
};

} // namespace axe