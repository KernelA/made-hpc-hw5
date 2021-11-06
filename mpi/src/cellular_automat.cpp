#include "mpi.h"
#include "stdafx.h"
#include "utils.h"

// #define DEBUG

const int NUM_GOST_CELLS{2};

void update_state(const std::vector<bool>& rule, std::vector<bool>& prev_stat,
                  std::vector<bool>& state) {
  std::vector<bool>::const_iterator current_value{prev_stat.cbegin() + 1};

  for (size_t i{1}; i < state.size() - 1; ++i) {
    short int rule_index{
        utils::bitmask2byte(current_value - 1, current_value + 2)};

    state.at(i) = rule.at(rule_index);
    ++current_value;
  }

  state.at(0) = state.at(1);
  state.at(state.size() - 1) = state.at(state.size() - 2);
}

void broadcast_rule(std::vector<bool>& rule, int root_rank, int rank) {
  int size{rule.size()};

  MPI_Bcast(&size, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

  bool temp_rules[size];

  for (size_t i{}; i < rule.size(); ++i) {
    temp_rules[i] = rule[i];
  }

  MPI_Bcast(temp_rules, size, MPI_CXX_BOOL, root_rank, MPI_COMM_WORLD);

  if (rank != root_rank) {
    for (size_t i{}; i < size; ++i) {
      rule.push_back(temp_rules[i]);
    }
  }
}

void send_cell_states(std::vector<bool>& cell_state, int block_size,
                      int reminder, int rank, int world_size) {
  size_t block_step{block_size + NUM_GOST_CELLS};

  const int SEND_SIZE = block_size + NUM_GOST_CELLS;

  bool buffer[SEND_SIZE];

  size_t start_index{block_step + reminder};

  for (int to_rank{}; to_rank < world_size; ++to_rank) {
    if (to_rank == rank) {
      continue;
    }

    for (int i{}; i < SEND_SIZE; ++i) {
      buffer[i] = cell_state.at(i + start_index);
    }

#ifdef DEBUG
    std::cout << "Send " << SEND_SIZE << std::endl;
#endif

    MPI_Send(&SEND_SIZE, 1, MPI_INT, to_rank, to_rank, MPI_COMM_WORLD);
    MPI_Send(buffer, SEND_SIZE, MPI_CXX_BOOL, to_rank, to_rank, MPI_COMM_WORLD);

    start_index += block_step;
  }
}

void rec_cell_state(std::vector<bool>& cell_state, int rank, int source_rank) {
  if (source_rank == rank) {
    return;
  }

  int size;
  MPI_Recv(&size, 1, MPI_INT, source_rank, MPI_ANY_TAG, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

#ifdef DEBUG
  std::cout << "Receive num states: " << size << std::endl;
#endif

  bool temp_cell_states[size];

  MPI_Recv(temp_cell_states, size, MPI_CXX_BOOL, source_rank, MPI_ANY_TAG,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for (int i{}; i < size; ++i) {
    cell_state.push_back(temp_cell_states[i]);
  }
}

void sync_states(std::vector<bool>& cell_stat, MPI_Comm comm, int world_rank) {
  int local_world_size = {};
  MPI_Comm_size(comm, &local_world_size);
  int local_rank{};
  MPI_Comm_rank(comm, &local_rank);

#ifdef DEBUG
  std::cout << "Global rank: " << world_rank << " local rank " << local_rank
            << " local word size " << local_world_size << std::endl;
#endif

  // Send right border
  if (local_rank + 1 < local_world_size && local_rank == 0) {
    bool right_bound = cell_stat.at(cell_stat.size() - 1);
    bool left_bound_from_remote;

    MPI_Send(&right_bound, 1, MPI_CXX_BOOL, local_rank + 1, local_rank + 1,
             comm);
    MPI_Recv(&left_bound_from_remote, 1, MPI_CXX_BOOL, local_rank + 1,
             MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
    cell_stat.at(cell_stat.size() - 1) = left_bound_from_remote;
  } else if (local_rank == 1) {
    bool left_bound_from_remote;
    bool left_bound = cell_stat.at(0);
    MPI_Recv(&left_bound_from_remote, 1, MPI_CXX_BOOL, local_rank - 1,
             MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);
    cell_stat.at(0) = left_bound_from_remote;

    MPI_Send(&left_bound, 1, MPI_CXX_BOOL, local_rank - 1, local_rank - 1,
             comm);
  }
}

void init_rule(std::vector<bool>& rules, int rule_num) {
  auto rule_binary_code{utils::byte2bitmask(rule_num)};

  for (auto it{rule_binary_code.crbegin()}; it != rule_binary_code.crend();
       ++it) {
    rules.push_back(*it);
  }
}

void init_cell_state(std::vector<bool>& cellular_state,
                     size_t num_cells_per_process, size_t reminder,
                     int world_size, bool is_period_condition) {
  cellular_state.clear();

  // each block have left and right gost cell
  size_t gost_cells{NUM_GOST_CELLS * world_size};

  size_t total_states =
      world_size * num_cells_per_process + reminder + gost_cells;

  std::default_random_engine engine;
  std::uniform_int_distribution dist(0, 1);

  for (size_t i{}; i < total_states; ++i) {
    cellular_state.push_back(static_cast<bool>(dist(engine)));
  }

  // Init gost cells
  size_t i{};

  cellular_state.at(0) = cellular_state.at(1);

  for (; i < num_cells_per_process + reminder + 2; ++i) {
    if (i == num_cells_per_process + reminder + 1) {
      cellular_state.at(i) = cellular_state.at(i - 1);
    }
  }
  size_t total{};

  for (; i < cellular_state.size(); ++i) {
    if (total == 0) {
      cellular_state.at(i) = cellular_state.at(i + 1);
    } else if (total > num_cells_per_process) {
      cellular_state.at(i) = cellular_state.at(i - 1);
      total = 0;
      continue;
    }
    ++total;
  }

  if (is_period_condition) {
    cellular_state.at(0) = cellular_state.at(cellular_state.size() - 1);
  }

  cellular_state.shrink_to_fit();
}

void init_comm_and_groups(MPI_Comm* communicators, MPI_Group* groups,
                          MPI_Group world_group, int world_size,
                          bool is_period_boundary) {
  int ranks[2] = {};

  for (size_t i{}; i < world_size - 1; ++i) {
    ranks[0] = i;
    ranks[1] = i + 1;
    if (MPI_SUCCESS != MPI_Group_incl(world_group, 2, ranks, &groups[i])) {
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_GROUP);
    }
    MPI_Comm_create_group(MPI_COMM_WORLD, groups[i], i, &communicators[i]);
  }

  int num_last_group = 1;
  int last_comm_index = world_size - 1;

  // period
  if (is_period_boundary) {
    ranks[0] = world_size - 1;
    ranks[1] = 0;
    num_last_group = 2;
    if (MPI_SUCCESS !=
        MPI_Group_incl(world_group, 2, ranks, &groups[last_comm_index])) {
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_GROUP);
    }

    if (MPI_SUCCESS != MPI_Comm_create_group(
                           MPI_COMM_WORLD, groups[last_comm_index],
                           last_comm_index, &communicators[last_comm_index])) {
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COMM);
    }

  } else {
    int ranks[1] = {world_size - 1};

    if (MPI_SUCCESS !=
        MPI_Group_incl(world_group, 1, ranks, &groups[last_comm_index])) {
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_GROUP);
    }

    if (MPI_SUCCESS != MPI_Comm_create_group(
                           MPI_COMM_WORLD, groups[last_comm_index],
                           last_comm_index, &communicators[last_comm_index])) {
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COMM);
    }
  }
}

void gather_state(std::vector<bool>& actual_state,
                  const std::vector<bool>& cellular_state_with_gost_cells,
                  int total_cells, int num_cell_per_process,
                  int cell_root_reminder, int process_rank, int root_rank,
                  int world_size) {
  int send_sizes[world_size] = {};
  int displacements[world_size] = {};

  int disp_step{num_cell_per_process + cell_root_reminder};

  send_sizes[0] = num_cell_per_process + cell_root_reminder;
  displacements[0] = 0;

  for (ssize_t i{1}; i < world_size; ++i) {
    send_sizes[i] = num_cell_per_process;
    displacements[i] = disp_step;
    disp_step += num_cell_per_process;
  }

  bool temp_all_states[total_cells] = {};
  bool send_buffer[send_sizes[process_rank]] = {};

  for (size_t i{1}; i < cellular_state_with_gost_cells.size() - 1; ++i) {
    send_buffer[i - 1] = cellular_state_with_gost_cells.at(i);
  }

  MPI_Gatherv(send_buffer, send_sizes[process_rank], MPI_CXX_BOOL,
              temp_all_states, send_sizes, displacements, MPI_CXX_BOOL,
              root_rank, MPI_COMM_WORLD);

  if (process_rank == root_rank) {
    actual_state.resize(total_cells);
    for (size_t i{}; i < total_cells; ++i) {
      actual_state.at(i) = temp_all_states[i];
    }
  }
}
int main(int argc, char** argv) {
  using std::cerr;
  using std::cout;
  using std::endl;
  using std::vector;

  MPI_Init(&argc, &argv);

  int process_rank{}, world_size{};

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

  const int ROOT_RANK = 0;

  if (process_rank == ROOT_RANK) {
    if (argc != 4 && argc != 5) {
      cerr << "Plesae specify cell_length, rule, num_steps and any value "
              "boundary condition. For example 10 110 10"
           << endl;
      cerr << "For the periodically boundary: 10 110 10 b" << endl;
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
    }
  }

  vector<bool> rules;

  bool is_period_boundary{argc == 5};

  int rule_num{std::stoi(argv[2])};

  if (process_rank == ROOT_RANK) {
    if (is_period_boundary) {
      cout << "Periodically boundary condition enabled" << endl;
    } else {
      cout << "Periodically boundary condition disabled" << endl;
    }

    if (rule_num < 0 || rule_num > 255) {
      cerr << "Rule num must be byte value: [0; 255]" << endl;
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
    }

    init_rule(rules, rule_num);
  }

  broadcast_rule(rules, ROOT_RANK, process_rank);

#ifdef DEBUG
  // cout << "Rule: ";
  // for (size_t i{}; i < rules.size(); ++i) {
  //   cout << ' ' << i << ' ' << rules[i];
  // }
#endif

  vector<bool> prev_cellular_state_per_process, cellular_state_per_process;
  vector<bool> all_cellular_state_per_step;

  int total_cells{std::stoi(argv[1])};
  int total_steps{std::stoi(argv[3])};

  if (total_cells < 1) {
    cerr << "Total cells must be positive but got: " << total_cells << endl;
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
  }

  if (total_steps < 1) {
    cerr << "Total steps must be positive but got: " << total_steps << endl;
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
  }

  if (total_cells < world_size) {
    cerr << "Number of processors must be greather than number of cells"
         << endl;
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
  }

  size_t num_cells_per_process{total_cells / world_size};
  size_t reminder{total_cells % world_size};

  if (process_rank == ROOT_RANK) {
    all_cellular_state_per_step.reserve(total_cells);

    init_cell_state(cellular_state_per_process, num_cells_per_process, reminder,
                    world_size, is_period_boundary);

#ifdef DEBUG
    {
      std::cout << "Init states:\n";
      for (size_t i{}; i < cellular_state_per_process.size(); ++i) {
        std::cout << cellular_state_per_process.at(i);
      }
      std::cout << std::endl;

      size_t i{};

      for (; i < num_cells_per_process + reminder + 2; ++i) {
        if (i == 0 || i == num_cells_per_process + reminder + 1) {
          cout << 'x';
        } else {
          cout << '|';
        }
      }

      size_t total{};

      for (; i < cellular_state_per_process.size(); ++i) {
        if (total == 0) {
          cout << 'x';
        } else if (total > num_cells_per_process) {
          cout << 'x';
          total = 0;
          continue;
        } else {
          cout << '|';
        }
        ++total;
      }
      cout << endl;
    }
#endif

    send_cell_states(cellular_state_per_process, num_cells_per_process,
                     reminder, process_rank, world_size);
    // Save only part state on root
    cellular_state_per_process.resize(num_cells_per_process + reminder +
                                      NUM_GOST_CELLS);
  } else {
    rec_cell_state(cellular_state_per_process, process_rank, ROOT_RANK);
  }

  std::ofstream file;

  if (process_rank == 0) {
    std::stringstream name;
    name << "state_" << std::to_string(rule_num) << ".txt";
    file.open(name.str());
  }

  int num_groups{world_size};

  MPI_Group world_group;

  MPI_Comm_group(MPI_COMM_WORLD, &world_group);

  MPI_Group groups[world_size] = {};
  MPI_Comm communicators[world_size] = {};

  init_comm_and_groups(communicators, groups, world_group, world_size,
                       is_period_boundary);

  for (size_t step{}; step < total_steps; ++step) {
    if (is_period_boundary && process_rank == 0) {
      sync_states(cellular_state_per_process, communicators[world_size - 1],
                  process_rank);
    }
    sync_states(cellular_state_per_process, communicators[process_rank],
                process_rank);

    if (process_rank > 0 && communicators[process_rank - 1] != MPI_COMM_NULL) {
      sync_states(cellular_state_per_process, communicators[process_rank - 1],
                  process_rank);
    }

#ifdef DEBUG
    cout << "State " << process_rank << ": ";
    for (size_t i{}; i < cellular_state_per_process.size(); ++i) {
      cout << cellular_state_per_process.at(i);
    }
    cout << endl;
#endif

    gather_state(all_cellular_state_per_step, cellular_state_per_process,
                 total_cells, num_cells_per_process, reminder, process_rank,
                 ROOT_RANK, world_size);

    if (process_rank == 0) {
      for (auto start{all_cellular_state_per_step.cbegin()};
           start != all_cellular_state_per_step.cend(); ++start) {
        file << *start;
      }

      file << endl;
    }

    prev_cellular_state_per_process = cellular_state_per_process;

    update_state(rules, prev_cellular_state_per_process,
                 cellular_state_per_process);

#ifdef DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if (process_rank == ROOT_RANK) {
      cout << "State per step: " << step << '\n';
      for (size_t i{}; i < all_cellular_state_per_step.size(); ++i) {
        cout << all_cellular_state_per_step.at(i);
      }
      cout << endl;
    }
#endif
  }

  if (process_rank == 0) {
    file.close();
  }

  for (size_t i{}; i < world_size; ++i) {
    MPI_Group_free(&groups[i]);

    if (communicators[i] != MPI_COMM_NULL) {
      MPI_Comm_free(&communicators[i]);
    }
  }

  MPI_Group_free(&world_group);
  MPI_Finalize();
  return 0;
}
