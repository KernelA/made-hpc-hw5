#include "utils.h"

short int utils::bitmask2byte(std::vector<bool>::const_iterator begin,
                              std::vector<bool>::const_iterator end) {
  short int value{};
  short int power_two{1};

  auto rbegin = std::make_reverse_iterator(end);
  auto rend = std::make_reverse_iterator(begin);

  for (auto bit_value{rbegin}; bit_value != rend; bit_value++) {
    value += power_two * static_cast<short int>(*bit_value);
    power_two *= 2;
  }

  return value;
}

std::vector<bool> utils::byte2bitmask(short int number) {
  int bit_check{1};

  const int NUM_BITS{8};
  std::vector<bool> bit_mask(NUM_BITS, false);

  for (int i{}; i < NUM_BITS; i++) {
    bit_mask[NUM_BITS - i - 1] = number & bit_check;
    number >>= 1;
  }

  return bit_mask;
}