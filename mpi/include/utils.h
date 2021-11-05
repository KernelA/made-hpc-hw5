#pragma once

namespace utils {
short int bitmask2byte(std::vector<bool>::const_iterator begin,
                       std::vector<bool>::const_iterator end);

std::vector<bool> byte2bitmask(short int number);

}  // namespace utils