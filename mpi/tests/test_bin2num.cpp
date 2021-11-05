#include "stdafx.h"
#include "utils.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Too small arguments" << std::endl;
    return 1;
  }

  short int answer{std::stoi(argv[1])};
  std::vector<bool> bit_mask;

  size_t start_index_mask{};

  while (argv[2][start_index_mask] != '\0') {
    bit_mask.push_back(argv[2][start_index_mask] == '1');
    ++start_index_mask;
  }

  assert(answer == utils::bitmask2byte(bit_mask.cbegin(), bit_mask.cend()));

  return 0;
}