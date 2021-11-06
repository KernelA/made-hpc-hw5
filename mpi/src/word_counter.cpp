#include <unicode/normlzr.h>
#include <unicode/regex.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <unicode/ustream.h>

#include "mpi.h"
#include "stdafx.h"

// #define DEBUG

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

void sync_dictionary(WordMap& map, int process_rank, int rec_from_rank,
                     int dest_rank) {
  if (process_rank == dest_rank) {
    CounterType total_remote_words{};
    MPI_Recv(&total_remote_words, 1, MPI_UINT64_T, rec_from_rank, MPI_ANY_TAG,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    CounterType raw_buffer_size{};
    CounterType remote_word_count{};

    for (CounterType i{}; i < total_remote_words; ++i) {
      MPI_Recv(&raw_buffer_size, 1, MPI_UINT64_T, rec_from_rank, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      char buffer[raw_buffer_size];

      MPI_Recv(buffer, raw_buffer_size, MPI_CHAR, rec_from_rank, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::string remote_word(buffer);

      MPI_Recv(&remote_word_count, 1, MPI_UINT64_T, rec_from_rank, MPI_ANY_TAG,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      auto it{map.find(remote_word)};

      if (it != map.end()) {
        remote_word_count += it->second;
      }

      map.insert_or_assign(remote_word, remote_word_count);
    }
  } else {
    CounterType total_words{map.size()};

    MPI_Send(&total_words, 1, MPI_UINT64_T, dest_rank, rec_from_rank,
             MPI_COMM_WORLD);

    for (const auto& word_count_pair : map) {
      CounterType buffer_size{word_count_pair.first.size()};
      MPI_Send(&buffer_size, 1, MPI_UINT64_T, dest_rank, rec_from_rank,
               MPI_COMM_WORLD);

      MPI_Send(word_count_pair.first.c_str(), buffer_size, MPI_CHAR, dest_rank,
               rec_from_rank, MPI_COMM_WORLD);

      MPI_Send(&word_count_pair.second, 1, MPI_UINT64_T, dest_rank,
               rec_from_rank, MPI_COMM_WORLD);
    }
  }
}

void save_top_n(std::ofstream& out_stream, const WordMap& word_counter,
                int top_n) {
  using PriorityItem = std::pair<CounterType, std::string>;

  std::priority_queue<PriorityItem, std::vector<PriorityItem>,
                      std::greater<PriorityItem>>
      top_n_words;

  for (const auto& word_count_pair : word_counter) {
    if (top_n_words.size() < top_n) {
      top_n_words.push(
          std::make_pair(word_count_pair.second, word_count_pair.first));
    } else {
      if (top_n_words.top().first < word_count_pair.second) {
        top_n_words.pop();
        top_n_words.push(
            std::make_pair(word_count_pair.second, word_count_pair.first));
      }
    }
  }

  std::vector<PriorityItem> order_by_count;

  while (!top_n_words.empty()) {
    order_by_count.push_back(top_n_words.top());
    top_n_words.pop();
  }

  for (auto it{order_by_count.crbegin()}; it != order_by_count.crend(); ++it) {
    out_stream << it->second << ' ' << it->first << std::endl;
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

  if (argc != 3) {
    cerr << "Specify basename to the file and top n words. Full name is "
            "basename<process_rank>.txt\nprocess_rank starts from zero"
         << endl;
    cout << "Example: basename 100" << endl;
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
  }

  int topN{std::stoi(argv[2])};

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
  cout << "Builded dict " << process_rank << endl;
#endif

  std::vector<int> communition_ranks;
  std::vector<int> new_communications;

  for (int i{}; i < world_size; ++i) {
    communition_ranks.push_back(i);
  }

  int local_rank = process_rank;
  int ranks[2] = {};

  while (communition_ranks.size() > 1) {
    int new_local_rank = local_rank / 2;

#ifdef DEBUG
    cout << "Tets " << new_local_rank << ' ' << process_rank << endl;
#endif
    int reminder = local_rank % 2;
    int size = 2;

    if (local_rank == communition_ranks.size() - 1 &&
        communition_ranks.size() % 2 == 1) {
      size = 1;
      ranks[0] = communition_ranks.at(new_local_rank);
    } else if (reminder == 1) {
      ranks[0] = communition_ranks.at(local_rank - 1);
      ranks[1] = communition_ranks.at(local_rank);
    } else {
      ranks[0] = communition_ranks.at(local_rank);
      ranks[1] = communition_ranks.at(local_rank + 1);
    }

#ifdef DEBUG
    cout << "Ranks " << process_rank << ' ' << ranks[0] << ' ' << ranks[1]
         << endl;
#endif

    if (size != 1) {
      sync_dictionary(word_counter, process_rank, ranks[1], ranks[0]);
    }

    if (reminder == 1) {
      word_counter.clear();
      break;
    }

    for (size_t i{}; i < communition_ranks.size(); i += 2) {
      new_communications.push_back(communition_ranks[i]);
    }

#ifdef DEBUG
    cout << "new comm ";
    for (const auto& rank : new_communications) {
      cout << ' ' << rank;
    }

    cout << endl;

#endif

    communition_ranks = new_communications;
    new_communications.clear();
    local_rank = new_local_rank;
  }

  if (!word_counter.empty()) {
    std::ofstream out("top_n.txt");

    save_top_n(out, word_counter, topN);
    out.close();
  }

#ifdef DEBUG
  cout << "Exited " << process_rank << endl;
#endif

  MPI_Finalize();

  return 0;
}