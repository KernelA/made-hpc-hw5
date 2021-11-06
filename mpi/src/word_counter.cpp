#include <unicode/normlzr.h>
#include <unicode/regex.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <unicode/ustream.h>

#include "mpi.h"
#include "stdafx.h"

#define DEBUG

const size_t MAX_WORDS_PER_LINE = 100'000;

using CounterType = std::uint64_t;
using SizeType = std::uint64_t;
using WordMap = std::unordered_map<std::string, CounterType>;

bool is_word(const icu::UnicodeString& string) {
  if (string.isEmpty()) {
    return false;
  }

  if (!u_isalpha(string.char32At(0))) {
    return false;
  }

  for (size_t i{1}; i < string.length() - 1; ++i) {
    auto letter = string.char32At(i);
    if (!u_isalpha(letter) &&
        !u_hasBinaryProperty(letter, UProperty::UCHAR_HYPHEN)) {
      return false;
    }
  }

  return u_isalpha(string.char32At(string.length() - 1));
}

CounterType total_lines(std::ifstream& stream) {
  std::string line;
  CounterType num_lines{};
  while (stream) {
    std::getline(stream, line);

    if (stream.eof()) {
      break;
    }
    ++num_lines;
  }

  return num_lines;
}

WordMap build_local_word_map(std::ifstream& stream) {
  using std::array;
  using std::cerr;
  using std::endl;
  using std::string;
  using namespace icu;

  string raw_text_line;
  UErrorCode status = U_ZERO_ERROR;

  RegexMatcher split_regex("\\s|\\p{punct}+",
                           URegexpFlag::UREGEX_CASE_INSENSITIVE, status);

  if (U_FAILURE(status)) {
    cerr << "Invalid regex expression" << endl;
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
  }

  UnicodeString line;
  array<UnicodeString, MAX_WORDS_PER_LINE> words;

  WordMap word_counter;

  status = U_ZERO_ERROR;

  auto normalizer = Normalizer2::getNFKDInstance(status);

  if (U_FAILURE(status)) {
    cerr << "Cannot get unicode normalizer" << endl;
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
  }

  status = U_ZERO_ERROR;

  while (stream) {
    std::getline(stream, raw_text_line);

    if (stream.eof()) {
      break;
    }

    line = normalizer->normalize(
        icu::UnicodeString::fromUTF8(raw_text_line).toLower().trim(), status);

    if (U_FAILURE(status)) {
      cerr << "cannot normalize unicode string" << endl;
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
    }

    if (!line.isEmpty()) {
      auto word_count =
          split_regex.split(line, words.data(), MAX_WORDS_PER_LINE, status);

      for (size_t i{}; i < word_count; ++i) {
        if (is_word(words[i])) {
          string raw_word;
          words[i].toUTF8String(raw_word);

          auto word_iter{word_counter.find(raw_word)};

          CounterType new_word_count{1};

          if (word_iter != word_counter.end()) {
            new_word_count = ++word_iter->second;
          }
          word_counter.insert_or_assign(raw_word, new_word_count);
        }
      }
    }
  }

  return word_counter;
}

void sync_line_blocks(CounterType& num_lines_per_process, int rank,
                      int root_rank) {
  MPI_Bcast(&num_lines_per_process, 1, MPI_UINT64_T, root_rank, MPI_COMM_WORLD);
}

void eval_total(std::vector<SizeType>& dict_sizes, const WordMap& map, int rank,
                int root_rank, int world_size) {
  dict_sizes.resize(world_size);

  auto dict_size = dict_sizes.size();

  MPI_Gather(&dict_size, 1, MPI_UINT64_T, dict_sizes.data(), world_size,
             MPI_UINT64_T, root_rank, MPI_COMM_WORLD);
}

void sync_dictionary(WordMap& map, int from_rank, int process_rank,
                     int dest_rank) {
  if (process_rank == dest_rank) {
    CounterType total_remote_words{};
    MPI_Recv(&total_remote_words, 1, MPI_UINT64_T, from_rank, MPI_ANY_TAG,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    CounterType raw_buffer_size{};
    CounterType remote_word_count{};

    for (CounterType i{}; i < total_remote_words; ++i) {
      MPI_Recv(&raw_buffer_size, 1, MPI_UINT64_T, from_rank, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      char buffer[raw_buffer_size];

      MPI_Recv(buffer, raw_buffer_size, MPI_CHAR, from_rank, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::string remote_word(buffer);

      MPI_Recv(&remote_word_count, 1, MPI_UINT64_T, from_rank, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      auto it{map.find(remote_word)};

      if (it != map.end()) {
        remote_word_count += it->second;
      }

      map.insert_or_assign(remote_word, remote_word_count);
    }
  } else {
    CounterType total_words{map.size()};

    MPI_Send(&total_words, 1, MPI_UINT64_T, dest_rank, from_rank,
             MPI_COMM_WORLD);

    for (const auto& word_count_pair : map) {
      CounterType buffer_size{word_count_pair.first.size()};
      MPI_Send(&buffer_size, 1, MPI_UINT64_T, dest_rank, from_rank,
               MPI_COMM_WORLD);

      MPI_Send(word_count_pair.first.c_str(), buffer_size, MPI_CHAR, dest_rank,
               from_rank, MPI_COMM_WORLD);

      MPI_Send(&word_count_pair.second, 1, MPI_UINT64_T, dest_rank, from_rank,
               MPI_COMM_WORLD);
    }
  }
}

int main(int argc, char** argv) {
  using std::cerr;
  using std::cout;
  using std::endl;
  using std::string;
  using namespace icu;

  const int ROOT_RANK = 0;

  MPI_Init(&argc, &argv);

  if (argc != 2) {
    cerr << "Specify basename to the file. Full name is "
            "basename<process_rank>.txt\nprocess_rank starts from zero"
         << endl;
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
  }

  int process_rank{}, world_size{};
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

  std::stringstream filename;
  filename << argv[1] << process_rank << ".txt";

  std::ifstream file(filename.str());

  if (!file.is_open()) {
    cerr << "Cannot open " << argv[1] << endl;
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_BAD_FILE);
  }

  auto word_counter = build_local_word_map(file);

  file.close();

#ifdef DEBUG
  cout << "Build dict " << process_rank << endl;
#endif

   int dest_rank = ROOT_RANK;

  if (process_rank == dest_rank) {
    for (int i{}; i < world_size; ++i) {
      if (i == dest_rank) {
        continue;
      }

      sync_dictionary(word_counter, i, process_rank, dest_rank);
    }
  } else {
    sync_dictionary(word_counter, process_rank, process_rank, dest_rank);
    word_counter.clear();
#ifdef DEBUG
    cout << "Sync own dict " << process_rank << endl;
#endif
  }

  if (process_rank == ROOT_RANK) {
  }

  MPI_Finalize();

  return 0;
}