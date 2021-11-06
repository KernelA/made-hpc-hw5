#include "mpi.h"
#include "perm.hpp"
#include "stdafx.h"

const int PROCESS_NAME_LENGTH = MPI_MAX_PROCESSOR_NAME;

using IndexType = unsigned int;

struct Contacts {
  int total_contacts;
  IndexType offset;
  int* ranks;
  char* names;
  int* next_contact;

  Contacts(int total_contacts)
      : total_contacts(total_contacts),
        offset(0),
        ranks(nullptr),
        names(nullptr),
        next_contact(nullptr) {
    ranks = new int[total_contacts];
    names = new char[total_contacts * PROCESS_NAME_LENGTH];
    next_contact = new int[total_contacts];
  }

  const char* name_at(IndexType index) const {
    return names + (index * PROCESS_NAME_LENGTH);
  }

  void init_send_order(const int* send_order) {
    for (size_t i{}; i < total_contacts; ++i) {
      next_contact[i] = send_order[i];
    }
  }

  int send_to() const { return next_contact[offset]; }

  void add_rank_name(int new_rank, const char* new_name) {
    ranks[offset] = new_rank;

    std::strncpy(names + offset * PROCESS_NAME_LENGTH, new_name,
                 PROCESS_NAME_LENGTH);

    offset++;
  }

  ~Contacts() {
    delete[] ranks;
    delete[] names;
  }

  size_t get_buffer_length() const {
    return sizeof(int) + sizeof(IndexType) + buffer_length_per_ranks() +
           buffer_size_per_names() + buffer_length_per_next_contact();
  }

  size_t buffer_length_per_next_contact() const {
    return sizeof(int) * total_contacts;
  }

  size_t buffer_length_per_ranks() const {
    return sizeof(int) * total_contacts;
  }

  size_t buffer_size_per_names() const {
    return sizeof(char) * total_contacts * PROCESS_NAME_LENGTH;
  }
};

void mpi_rec_contacts(Contacts& contacts) {
  size_t buffer_size{contacts.get_buffer_length()};

  char* buffer = new char[buffer_size];

  MPI_Status status;

  MPI_Recv(buffer, buffer_size, MPI_PACKED, MPI_ANY_SOURCE, MPI_ANY_TAG,
           MPI_COMM_WORLD, &status);

  int position{};

  MPI_Unpack(buffer, buffer_size, &position, &contacts.total_contacts, 1,
             MPI_INT, MPI_COMM_WORLD);

  MPI_Unpack(buffer, buffer_size, &position, &contacts.offset, 1, MPI_UNSIGNED,
             MPI_COMM_WORLD);

  MPI_Unpack(buffer, buffer_size, &position, contacts.ranks,
             contacts.total_contacts, MPI_INT, MPI_COMM_WORLD);

  MPI_Unpack(buffer, buffer_size, &position, contacts.names,
             contacts.total_contacts * PROCESS_NAME_LENGTH, MPI_CHAR,
             MPI_COMM_WORLD);

  MPI_Unpack(buffer, buffer_size, &position, contacts.next_contact,
             contacts.total_contacts, MPI_INT, MPI_COMM_WORLD);

  delete[] buffer;
}

void mpi_send_contacts(const Contacts& contacts, int dest_rank) {
  size_t buffer_size{contacts.get_buffer_length()};
  char* buffer = new char[buffer_size];

  int position{};

  MPI_Pack(&contacts.total_contacts, 1, MPI_INT, buffer, buffer_size, &position,
           MPI_COMM_WORLD);

  MPI_Pack(&contacts.offset, 1, MPI_UNSIGNED, buffer, buffer_size, &position,
           MPI_COMM_WORLD);

  MPI_Pack(contacts.ranks, contacts.total_contacts, MPI_INT, buffer,
           buffer_size, &position, MPI_COMM_WORLD);

  MPI_Pack(contacts.names, contacts.total_contacts * PROCESS_NAME_LENGTH,
           MPI_CHAR, buffer, buffer_size, &position, MPI_COMM_WORLD);

  MPI_Pack(contacts.next_contact, contacts.total_contacts, MPI_INT, buffer,
           buffer_size, &position, MPI_COMM_WORLD);

  MPI_Ssend(buffer, buffer_size, MPI_PACKED, dest_rank, dest_rank,
            MPI_COMM_WORLD);

  delete[] buffer;
}

std::ostream& operator<<(std::ostream& stream, const Contacts& contacts) {
  stream << "Prev contacts\n Rank\tName" << std::endl;
  for (IndexType i{}; i < contacts.offset; i++) {
    stream << contacts.ranks[i] << '\t' << contacts.name_at(i) << std::endl;
  }

  return stream;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int process_rank{}, world_size{};

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

  if (process_rank == 0) {
    Contacts contacts(world_size);

    {
      std::default_random_engine engine;

      int send_order[world_size];

      for (int i{}; i < world_size; ++i) {
        send_order[i] = i;
      }

      comb::random_permutation(send_order, world_size, engine);

      for (int i{}; i < world_size; ++i) {
        if (send_order[i] == 0) {
          std::swap(send_order[i], send_order[world_size - 1]);
          break;
        }
      }

      contacts.init_send_order(send_order);
    }

    int next_rank{contacts.send_to()};
    std::stringstream string_stream;
    string_stream << "process" << process_rank;
    contacts.add_rank_name(process_rank, string_stream.str().c_str());

    mpi_send_contacts(contacts, next_rank);
    mpi_rec_contacts(contacts);

    std::cout << "Communication world:\n" << contacts;

  } else if (process_rank != 0) {
    Contacts contacts(world_size);
    mpi_rec_contacts(contacts);
    int next_rank{contacts.send_to()};
    std::stringstream string_stream;
    string_stream << "process" << process_rank;
    contacts.add_rank_name(process_rank, string_stream.str().c_str());
    mpi_send_contacts(contacts, next_rank);
  }

  MPI_Finalize();
  return 0;
}